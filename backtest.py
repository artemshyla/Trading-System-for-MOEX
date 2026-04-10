from __future__ import annotations

import os
import copy
import random
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import randint, uniform

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV, RandomizedSearchCV, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

import moexalgo
from moexalgo import Ticker
import moexalgo.engines.currency
from moexalgo import Market


def _to_simple_ret(ret, ret_type='log'):
    ret = np.asarray(ret, dtype=float)

    if ret_type == 'log':
        return np.expm1(ret)
    elif ret_type == 'simple':
        return ret
    else:
        raise ValueError("ret_type must be 'log' or 'simple'")


def _calc_bt_metrics(net_ret, turnover, pos, equity, drawdown, periods_per_year=252):
    net_ret = np.asarray(net_ret, dtype=float)
    turnover = np.asarray(turnover, dtype=float)
    pos = np.asarray(pos)
    equity = np.asarray(equity, dtype=float)
    drawdown = np.asarray(drawdown, dtype=float)

    n = len(net_ret)
    if n == 0:
        return {
            'total_return': 0.0,
            'cagr': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'vol': 0.0,
            'max_drawdown': 0.0,
            'trades': 0,
            'turnover_mean': 0.0,
            'exposure': 0.0,
            'hit_rate': 0.0,
            'active_hit_rate': 0.0,
            'avg_daily_ret': 0.0,
        }

    mean_ret = net_ret.mean()
    std_ret = net_ret.std(ddof=1) if n > 1 else 0.0

    downside = net_ret[net_ret < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0

    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0
    sortino = (mean_ret / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0
    vol = std_ret * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

    total_return = equity[-1] - 1.0
    cagr = equity[-1] ** (periods_per_year / n) - 1.0 if equity[-1] > 0 else -1.0

    active_mask = pos != 0
    active_ret = net_ret[active_mask]

    hit_rate = (net_ret > 0).mean()
    active_hit_rate = (active_ret > 0).mean() if len(active_ret) > 0 else 0.0

    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'vol': float(vol),
        'max_drawdown': float(drawdown.min()),
        'trades': int((turnover > 0).sum()),
        'turnover_mean': float(turnover.mean()),
        'exposure': float((active_mask).mean()),
        'hit_rate': float(hit_rate),
        'active_hit_rate': float(active_hit_rate),
        'avg_daily_ret': float(mean_ret),
    }

def _normalize_signal_ranges(x, name):
    if x is None:
        return {"mode": "ranges", "ranges": []}

    if np.isscalar(x):
        return {"mode": "scalar", "value": float(x)}

    arr = np.asarray(x, dtype=float)

    if arr.ndim == 1:
        if len(arr) == 0:
            return {"mode": "ranges", "ranges": []}
        if len(arr) != 2:
            raise ValueError(f"{name} must be a scalar or list of [lo, hi] pairs")
        arr = arr.reshape(1, 2)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be a scalar or list of [lo, hi] pairs")

    ranges = []
    for lo, hi in arr:
        lo = float(lo)
        hi = float(hi)

        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"{name} contains non-finite values")

        if lo >= hi:
            raise ValueError(f"Each range in {name} must satisfy lo < hi")

        ranges.append((lo, hi))

    ranges = sorted(ranges, key=lambda z: z[0])

    for i in range(1, len(ranges)):
        prev_lo, prev_hi = ranges[i - 1]
        cur_lo, cur_hi = ranges[i]
        if cur_lo < prev_hi:
            raise ValueError(f"Ranges inside {name} overlap: {(prev_lo, prev_hi)} and {(cur_lo, cur_hi)}")

    return {"mode": "ranges", "ranges": ranges}


def _mask_from_rule(values, rule, side_name):
    values = np.asarray(values, dtype=float)
    mask = np.zeros(len(values), dtype=bool)
    labels = np.full(len(values), "", dtype=object)

    if rule["mode"] == "scalar":
        thr = rule["value"]

        if side_name == "upper":
            mask = values > thr
            labels[mask] = f"upper>{thr:.6f}"
        elif side_name == "lower":
            mask = values < thr
            labels[mask] = f"lower<{thr:.6f}"
        else:
            raise ValueError("side_name must be 'upper' or 'lower'")

        return mask, labels

    for i, (lo, hi) in enumerate(rule["ranges"], 1):
        cur = (values > lo) & (values <= hi)
        mask |= cur
        labels[cur] = f"{side_name}_{i}:({lo:.6f},{hi:.6f}]"

    return mask, labels


def backtest_probs(proba_long,
                   fwd_ret,
                   date=None,
                   lower=0.45,
                   upper=0.55,
                   cost_bps=10,
                   ret_type='log',
                   periods_per_year=252):
    proba_long = np.asarray(proba_long, dtype=float)
    fwd_ret = np.asarray(fwd_ret, dtype=float)

    if len(proba_long) != len(fwd_ret):
        raise ValueError('len(proba_long) must be equal to len(fwd_ret)')

    if date is None:
        date = np.arange(len(proba_long))
    else:
        date = pd.to_datetime(date)

    lower_rule = _normalize_signal_ranges(lower, "lower")
    upper_rule = _normalize_signal_ranges(upper, "upper")

    if lower_rule["mode"] == "scalar" and upper_rule["mode"] == "scalar":
        if lower_rule["value"] >= upper_rule["value"]:
            raise ValueError("For scalar thresholds, lower must be < upper")

    long_mask, long_labels = _mask_from_rule(proba_long, upper_rule, "upper")
    short_mask, short_labels = _mask_from_rule(proba_long, lower_rule, "lower")

    overlap = long_mask & short_mask
    if overlap.any():
        overlap_probs = np.unique(np.round(proba_long[overlap], 8))
        raise ValueError(f"Long/short zones overlap for some probabilities: {overlap_probs[:10]}")

    fwd_ret_simple = _to_simple_ret(fwd_ret, ret_type=ret_type)

    pos = np.zeros(len(proba_long), dtype=np.int8)
    pos[long_mask] = 1
    pos[short_mask] = -1

    signal_zone = np.full(len(proba_long), "flat", dtype=object)
    signal_zone[long_mask] = long_labels[long_mask]
    signal_zone[short_mask] = short_labels[short_mask]

    pos_prev = np.r_[0, pos[:-1]]
    turnover = np.abs(pos - pos_prev).astype(float)

    cost_rate = cost_bps / 10000.0
    cost = turnover * cost_rate

    gross_ret = pos * fwd_ret_simple
    net_ret = gross_ret - cost

    equity = np.cumprod(1.0 + net_ret)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    bt = pd.DataFrame({
        'date': date,
        'proba_long': proba_long,
        'fwd_ret': fwd_ret,
        'fwd_ret_simple': fwd_ret_simple,
        'signal_zone': signal_zone,
        'pos': pos,
        'pos_prev': pos_prev,
        'turnover': turnover,
        'cost': cost,
        'gross_ret': gross_ret,
        'net_ret': net_ret,
        'equity': equity,
        'peak': peak,
        'drawdown': drawdown,
    })

    metrics = _calc_bt_metrics(
        net_ret=bt['net_ret'].values,
        turnover=bt['turnover'].values,
        pos=bt['pos'].values,
        equity=bt['equity'].values,
        drawdown=bt['drawdown'].values,
        periods_per_year=periods_per_year
    )

    yearly = None
    if 'date' in bt.columns and np.issubdtype(bt['date'].dtype, np.datetime64):
        yearly_rows = []

        for year, g in bt.groupby(bt['date'].dt.year):
            eq = (1.0 + g['net_ret']).cumprod().values
            peak_y = np.maximum.accumulate(eq)
            dd_y = eq / peak_y - 1.0

            m = _calc_bt_metrics(
                net_ret=g['net_ret'].values,
                turnover=g['turnover'].values,
                pos=g['pos'].values,
                equity=eq,
                drawdown=dd_y,
                periods_per_year=periods_per_year
            )
            m['year'] = int(year)
            yearly_rows.append(m)

        yearly = pd.DataFrame(yearly_rows)
        if len(yearly) > 0:
            yearly = yearly[['year', 'total_return', 'cagr', 'sharpe', 'sortino',
                             'vol', 'max_drawdown', 'trades', 'turnover_mean',
                             'exposure', 'hit_rate', 'active_hit_rate', 'avg_daily_ret']]

    return {
        'metrics': metrics,
        'bt': bt,
        'yearly': yearly
    }

def print_bt_report(bt_res):
    m = bt_res['metrics']

    print(f"total_return:    {m['total_return']:.4f}")
    print(f"cagr:            {m['cagr']:.4f}")
    print(f"sharpe:          {m['sharpe']:.4f}")
    print(f"sortino:         {m['sortino']:.4f}")
    print(f"vol:             {m['vol']:.4f}")
    print(f"max_drawdown:    {m['max_drawdown']:.4f}")
    print(f"trades:          {m['trades']}")
    print(f"turnover_mean:   {m['turnover_mean']:.4f}")
    print(f"exposure:        {m['exposure']:.4f}")
    print(f"hit_rate:        {m['hit_rate']:.4f}")
    print(f"active_hit_rate: {m['active_hit_rate']:.4f}")


def plot_bt(bt_res, figsize=(14, 5)):
    bt = bt_res['bt']

    plt.figure(figsize=figsize)
    plt.plot(bt['date'], bt['equity'], label='equity')
    plt.title('Backtest equity')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=figsize)
    plt.plot(bt['date'], bt['drawdown'], label='drawdown')
    plt.title('Backtest drawdown')
    plt.grid()
    plt.legend()
    plt.show()

def confidence_report(df, probs, label="valid", n_bins=10):
    out = df[["date", "close", "fwd_ret", "y"]].copy().reset_index(drop=True)
    out["proba"] = np.asarray(probs, dtype=float)
    out["score"] = 2.0 * out["proba"] - 1.0
    out["confidence"] = np.abs(out["score"])
    out["pred"] = (out["proba"] >= 0.5).astype(int)
    out["pred_side"] = np.where(out["pred"] == 1, 1.0, -1.0)
    out["next_ret_simple"] = np.expm1(out["fwd_ret"])
    out["next_price_diff"] = out["close"] * out["next_ret_simple"]
    out["aligned_ret"] = out["pred_side"] * out["next_ret_simple"]
    out["correct"] = (out["pred"] == out["y"]).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    q1 = min(n_bins, out["proba"].nunique())
    q2 = min(n_bins, out["confidence"].nunique())

    out["proba_bin"] = pd.qcut(out["proba"], q=q1, duplicates="drop")
    out["confidence_bin"] = pd.qcut(out["confidence"], q=q2, duplicates="drop")

    proba_table = (
        out.groupby("proba_bin", observed=False)
        .agg(
            n=("proba", "size"),
            mean_proba=("proba", "mean"),
            mean_score=("score", "mean"),
            pos_rate=("y", "mean"),
            accuracy=("correct", "mean"),
            mean_next_ret=("next_ret_simple", "mean"),
            median_next_ret=("next_ret_simple", "median"),
            mean_next_price_diff=("next_price_diff", "mean"),
            mean_aligned_ret=("aligned_ret", "mean"),
            std_aligned_ret=("aligned_ret", "std")
        )
        .reset_index()
    )

    proba_table["mean_next_ret_bp"] = proba_table["mean_next_ret"] * 10000
    proba_table["median_next_ret_bp"] = proba_table["median_next_ret"] * 10000
    proba_table["mean_aligned_ret_bp"] = proba_table["mean_aligned_ret"] * 10000

    confidence_table = (
        out.groupby("confidence_bin", observed=False)
        .agg(
            n=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            mean_abs_next_ret=("next_ret_simple", lambda x: np.mean(np.abs(x))),
            mean_next_ret=("next_ret_simple", "mean"),
            mean_aligned_ret=("aligned_ret", "mean"),
            median_aligned_ret=("aligned_ret", "median"),
            mean_next_price_diff=("next_price_diff", "mean")
        )
        .reset_index()
    )

    confidence_table["mean_abs_next_ret_bp"] = confidence_table["mean_abs_next_ret"] * 10000
    confidence_table["mean_next_ret_bp"] = confidence_table["mean_next_ret"] * 10000
    confidence_table["mean_aligned_ret_bp"] = confidence_table["mean_aligned_ret"] * 10000
    confidence_table["median_aligned_ret_bp"] = confidence_table["median_aligned_ret"] * 10000

    stats = pd.Series({
        "label": label,
        "pearson_proba_vs_next_ret": out["proba"].corr(out["next_ret_simple"], method="pearson"),
        "spearman_proba_vs_next_ret": out["proba"].corr(out["next_ret_simple"], method="spearman"),
        "pearson_score_vs_next_ret": out["score"].corr(out["next_ret_simple"], method="pearson"),
        "spearman_score_vs_next_ret": out["score"].corr(out["next_ret_simple"], method="spearman"),
        "pearson_conf_vs_abs_ret": out["confidence"].corr(out["next_ret_simple"].abs(), method="pearson"),
        "spearman_conf_vs_abs_ret": out["confidence"].corr(out["next_ret_simple"].abs(), method="spearman"),
        "pearson_conf_vs_aligned_ret": out["confidence"].corr(out["aligned_ret"], method="pearson"),
        "spearman_conf_vs_aligned_ret": out["confidence"].corr(out["aligned_ret"], method="spearman"),
        "overall_accuracy": out["correct"].mean(),
        "overall_mean_next_ret_bp": out["next_ret_simple"].mean() * 10000,
        "overall_mean_aligned_ret_bp": out["aligned_ret"].mean() * 10000
    })

    return {
        "raw": out,
        "proba_table": proba_table,
        "confidence_table": confidence_table,
        "stats": stats
    }