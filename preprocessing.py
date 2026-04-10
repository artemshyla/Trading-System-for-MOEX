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


def _download_feature(ticker, start_date, end_date, period):
    tick = Ticker(ticker)
    data = tick.candles(start=start_date, end=end_date, period=period)
    
    data['begin'] = pd.to_datetime(data['begin'])
    data['end'] = pd.to_datetime(data['end'])
    data['date'] = data['begin'].dt.floor('D')
    
    data_final = (
        data[['date', 'close']]
        .copy()
        .sort_values('date')
        .drop_duplicates('date')
        .reset_index(drop=True)
        .rename(columns={'close': f'{ticker.lower()}_close'})
    )
    return data_final

def _usdrub(start_date, end_date, period):
    usdrub = Ticker('USD000UTSTOM')
    df_usdrub = usdrub.candles(start=start_date, end='2022-04-30', period=period)
    df_usdrub['begin'] = pd.to_datetime(df_usdrub['begin'])
    df_usdrub['end'] = pd.to_datetime(df_usdrub['end'])
    df_usdrub['date'] = df_usdrub['begin'].dt.floor('D')
    
    df_usdrub_spot_small = (
        df_usdrub[['date', 'close']]
        .copy()
        .rename(columns={'close': 'usdrub_close_spot'})
        .sort_values('date')
        .drop_duplicates('date')
        .reset_index(drop=True)
    )

    usdrubf = Ticker('USDRUBF')
    df_usdrubf = usdrubf.candles(start='2022-04-20', end=end_date, period=period)
    df_usdrubf['begin'] = pd.to_datetime(df_usdrubf['begin'])
    df_usdrubf['end'] = pd.to_datetime(df_usdrubf['end'])
    df_usdrubf['date'] = df_usdrubf['begin'].dt.floor('D')
    
    df_usdrub_fut_small = (
        df_usdrubf[['date', 'close']]
        .copy()
        .rename(columns={'close': 'usdrub_close_fut'})
        .sort_values('date')
        .drop_duplicates('date')
        .reset_index(drop=True)
    )

    df_usdrub_all = (
        df_usdrub_spot_small
        .merge(df_usdrub_fut_small, on='date', how='outer')
        .sort_values('date')
        .reset_index(drop=True)
    )
    
    df_usdrub_all['usdrub_close'] = df_usdrub_all['usdrub_close_fut'].combine_first(
        df_usdrub_all['usdrub_close_spot']
    )
    df_usdrub_all = df_usdrub_all[['date', 'usdrub_close']].copy()
    df_usdrub_all = df_usdrub_all.drop_duplicates('date').reset_index(drop=True)
    
    df_usdrub_ret = df_usdrub_all.copy()
    return df_usdrub_ret

def _stock(ticker, start_date, end_date, period):
    stock = Ticker(ticker)
    data = stock.candles(start=start_date, end=end_date, period=period)
    data['begin'] = pd.to_datetime(data['begin'])
    data['date'] = data['begin'].dt.floor('D')
    data = (
        data[["date", "open", "high", "low", "close", "value", "volume"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return data

def _concat(data, *feature_dfs):
    data = (
        data.copy()
        .assign(date=pd.to_datetime(data['date']).dt.floor('D'))
        .sort_values('date')
        .drop_duplicates('date')
        .reset_index(drop=True)
    )

    rows_before = len(data)
    common_dates = set(data['date'])

    prepared_dfs = []
    for df in feature_dfs:
        cur = (
            df.copy()
            .assign(date=pd.to_datetime(df['date']).dt.floor('D'))
            .sort_values('date')
            .drop_duplicates('date')
            .reset_index(drop=True)
        )
        prepared_dfs.append(cur)
        common_dates &= set(cur['date'])

    common_dates = sorted(common_dates)

    result = data[data['date'].isin(common_dates)].copy()
    result = result.sort_values('date').reset_index(drop=True)

    for df in prepared_dfs:
        df = df[df['date'].isin(common_dates)].copy()
        df = df.sort_values('date').reset_index(drop=True)
        result = result.merge(df, on='date', how='inner')

    close_cols = [col for col in result.columns if col.endswith('_close')]

    for col in close_cols:
        ret_col = col.replace('_close', '_ret')
        result[ret_col] = np.log(result[col] / result[col].shift(1))

    rows_after = len(result)
    print(f'потеряно {rows_before - rows_after} - осталось {rows_after}')

    return result

def rolling_zscore(x, window):
    return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-9)

def rolling_beta(y, x, window):
    return y.rolling(window).cov(x) / (x.rolling(window).var() + 1e-9)

def _basic_features(data):
    data['r1']  = np.log(data['close'] / data['close'].shift(1))
    data['r3']  = np.log(data['close'] / data['close'].shift(3))
    data['r10']  = np.log(data['close'] / data['close'].shift(10))
    data['r20'] = np.log(data['close'] / data['close'].shift(20))
    data['r1_vol5']  = data['r1'].rolling(5).std()
    data['r1_vol20'] = data['r1'].rolling(20).std()
    data['range'] = (data['high'] - data['low']) / data['close']
    data['logvol'] = np.log1p(data['volume'])
    data["body"] = (data["close"] - data["open"]) / (data["open"] + 1e-9)
    data["upper_wick"] = (data["high"] - np.maximum(data["open"], data["close"])) / (data["open"] + 1e-9)
    data["lower_wick"] = (np.minimum(data["open"], data["close"]) - data["low"]) / (data["open"] + 1e-9)
    return data

def _context_features(data, lag_steps, mom_windows, vol_windows, context_tickers=None, benchmark='imoex'):
    if context_tickers is None:
        context_ret_cols = sorted([col for col in data.columns if col.endswith('_ret')])
    else:
        context_ret_cols = []
        for ticker in context_tickers:
            col = ticker if ticker.endswith('_ret') else f'{ticker.lower()}_ret'
            if col in data.columns:
                context_ret_cols.append(col)

    benchmark_col = benchmark if benchmark.endswith('_ret') else f'{benchmark.lower()}_ret'

    data['alpha_1d'] = data['r1'] - data[benchmark_col]

    for col in context_ret_cols:
        prefix = col.replace('_ret', '')

        for l in lag_steps:
            data[f'{prefix}_ret_lag{l}'] = data[col].shift(l)

        for w in mom_windows:
            data[f'{prefix}_mom{w}'] = data[col].rolling(w).sum()

        for w in vol_windows:
            data[f'{prefix}_vol{w}'] = data[col].rolling(w).std()

    for l in lag_steps:
        data[f'alpha_1d_lag{l}'] = data['alpha_1d'].shift(l)

    for w in mom_windows:
        data[f'alpha_mom{w}'] = data['alpha_1d'].rolling(w).sum()
        data[f'stock_mom{w}'] = data['r1'].rolling(w).sum()

    for w in vol_windows:
        data[f'alpha_vol{w}'] = data['alpha_1d'].rolling(w).std()
        data[f'stock_vol{w}'] = data['r1'].rolling(w).std()

    benchmark_name = benchmark_col.replace('_ret', '')

    data[f'beta_{benchmark_name}_10'] = rolling_beta(data['r1'], data[benchmark_col], 10)
    data[f'beta_{benchmark_name}_20'] = rolling_beta(data['r1'], data[benchmark_col], 20)

    data[f'corr_{benchmark_name}_10'] = data['r1'].rolling(10).corr(data[benchmark_col])
    data[f'corr_{benchmark_name}_20'] = data['r1'].rolling(20).corr(data[benchmark_col])

    data['rel_strength_5'] = data['r1'].rolling(5).sum() - data[benchmark_col].rolling(5).sum()
    data['rel_strength_10'] = data['r1'].rolling(10).sum() - data[benchmark_col].rolling(10).sum()
    data['rel_strength_20'] = data['r1'].rolling(20).sum() - data[benchmark_col].rolling(20).sum()

    return data

def _flow_trades(data, lag_steps, mom_windows, vol_windows):
    data["vwap"] = data["value"] / (data["volume"] + 1e-9)
    data["ts_close_to_vwap"] = (data["close"] / data["vwap"]) - 1
    data["ts_bar_return"] = (data["close"] / data["open"]) - 1
    data["ts_range_norm"] = (data["high"] - data["low"]) / data["close"]
    data["ts_turnover_per_vol"] = data["value"] / (data["volume"] + 1e-9)
    data["ts_log_turnover"] = np.log1p(data["value"])
    data["ts_volume_z"] = (
        data["volume"] - data["volume"].rolling(20).mean()
    ) / (data["volume"].rolling(20).std() + 1e-9)
    data["ts_vwap_mom"] = np.log(data["vwap"] / data["vwap"].shift(1))
    
    data["rel_volume"] = data["volume"] / data["volume"].rolling(20).mean()
    data["vol_ratio"] = data["r1_vol5"] / data["r1_vol20"]

    for l in lag_steps:
        data[f'ts_close_to_vwap_lag{l}'] = data['ts_close_to_vwap'].shift(l)
        data[f'ts_turnover_per_vol_lag{l}'] = data['ts_turnover_per_vol'].shift(l)
        data[f'ts_log_turnover_lag{l}'] = data['ts_log_turnover'].shift(l)
        data[f'ts_volume_z_lag{l}'] = data['ts_volume_z'].shift(l)
        data[f'ts_vwap_mom_lag{l}'] = data['ts_vwap_mom'].shift(l)

    data = data.copy()
    for w in [3, 5, 10, 20]:
        data[f'ts_close_to_vwap_ma{w}'] = data['ts_close_to_vwap'].rolling(w).mean()
        data[f'ts_close_to_vwap_z{w}'] = rolling_zscore(data['ts_close_to_vwap'], w)
    
        data[f'ts_turnover_per_vol_ma{w}'] = data['ts_turnover_per_vol'].rolling(w).mean()
        data[f'ts_turnover_per_vol_z{w}'] = rolling_zscore(data['ts_turnover_per_vol'], w)
        data[f'ts_turnover_per_vol_rel{w}'] = (
            data['ts_turnover_per_vol'] / (data['ts_turnover_per_vol'].rolling(w).mean() + 1e-9)
        ) - 1
    
        data[f'ts_log_turnover_z{w}'] = rolling_zscore(data['ts_log_turnover'], w)
        data[f'ts_volume_z_ma{w}'] = data['ts_volume_z'].rolling(w).mean()
        data[f'ts_vwap_mom_ma{w}'] = data['ts_vwap_mom'].rolling(w).mean()

    data['ts_turnover_per_vol_mom3'] = np.log(
        data['ts_turnover_per_vol'] / (data['ts_turnover_per_vol'].shift(3) + 1e-9)
    )
    data['ts_turnover_per_vol_mom5'] = np.log(
        data['ts_turnover_per_vol'] / (data['ts_turnover_per_vol'].shift(5) + 1e-9)
    )
    return data


def _regime_features(data, benchmark='imoex'):
    benchmark_col = benchmark if benchmark.endswith('_ret') else f'{benchmark.lower()}_ret'
    benchmark_name = benchmark_col.replace('_ret', '')

    data['ret_5d'] = data['r1'].rolling(5).sum()
    data['ret_10d'] = data['r1'].rolling(10).sum()
    data['ret_20d'] = data['r1'].rolling(20).sum()

    data['range_mean_5'] = data['range'].rolling(5).mean()
    data['range_mean_10'] = data['range'].rolling(10).mean()
    data['range_mean_20'] = data['range'].rolling(20).mean()

    data['range_z_20'] = (
        (data['range'] - data['range'].rolling(20).mean()) /
        data['range'].rolling(20).std()
    )

    data['trend_to_vol_10'] = data['ret_10d'] / data['r1'].rolling(10).std()
    data['trend_to_vol_20'] = data['ret_20d'] / data['r1'].rolling(20).std()

    data[f'{benchmark_name}_trend_to_vol_10'] = (
        data[benchmark_col].rolling(10).sum() / data[benchmark_col].rolling(10).std()
    )
    data[f'{benchmark_name}_trend_to_vol_20'] = (
        data[benchmark_col].rolling(20).sum() / data[benchmark_col].rolling(20).std()
    )

    data['alpha_to_vol_10'] = data['alpha_1d'].rolling(10).sum() / data['alpha_1d'].rolling(10).std()
    data['alpha_to_vol_20'] = data['alpha_1d'].rolling(20).sum() / data['alpha_1d'].rolling(20).std()

    data['vol_ratio_5_20'] = data['r1'].rolling(5).std() / data['r1'].rolling(20).std()
    data[f'{benchmark_name}_vol_ratio_5_20'] = (
        data[benchmark_col].rolling(5).std() / data[benchmark_col].rolling(20).std()
    )

    data['compression_10'] = data['range'].rolling(10).mean() / data['r1'].rolling(10).std()
    data['compression_20'] = data['range'].rolling(20).mean() / data['r1'].rolling(20).std()

    return data

def get_feature_groups(lag_steps, mom_windows, vol_windows, context_tickers, benchmark='imoex'):
    base_feats = [
        "r1", "r3", "r10", "r20",
        "r1_vol5", "r1_vol20",
        "range",
        "logvol",
        "body",
        "upper_wick", "lower_wick"
    ]

    context_names = [
        t.lower().replace('_ret', '').replace('_close', '')
        for t in context_tickers
    ]

    benchmark_name = benchmark.lower().replace('_ret', '').replace('_close', '')

    context_feats = [f"{name}_ret" for name in context_names] + ["alpha_1d"]

    flow_feats = [
        "ts_close_to_vwap",
        "ts_bar_return",
        "ts_range_norm",
        "ts_turnover_per_vol",
        "ts_log_turnover",
        "ts_volume_z",
        "ts_vwap_mom"
    ]

    regime_feats = [
        "rel_volume", "vol_ratio"
    ]

    extra_context_feats = (
        [f'{name}_ret_lag{l}' for name in context_names for l in lag_steps] +
        [f'{name}_mom{w}' for name in context_names for w in mom_windows] +
        [f'{name}_vol{w}' for name in context_names for w in vol_windows] +
        [f'alpha_1d_lag{l}' for l in lag_steps] +
        [f'alpha_mom{w}' for w in mom_windows] +
        [f'alpha_vol{w}' for w in vol_windows] +
        [f'stock_mom{w}' for w in mom_windows] +
        [f'stock_vol{w}' for w in vol_windows] +
        [
            f'beta_{benchmark_name}_10', f'beta_{benchmark_name}_20',
            f'corr_{benchmark_name}_10', f'corr_{benchmark_name}_20',
            'rel_strength_5', 'rel_strength_10', 'rel_strength_20'
        ]
    )

    extra_flow_feats = (
        [f'ts_close_to_vwap_lag{l}' for l in lag_steps] +
        [f'ts_turnover_per_vol_lag{l}' for l in lag_steps] +
        [f'ts_log_turnover_lag{l}' for l in lag_steps] +
        [f'ts_volume_z_lag{l}' for l in lag_steps] +
        [f'ts_vwap_mom_lag{l}' for l in lag_steps] +
        [f'ts_close_to_vwap_ma{w}' for w in [3, 5, 10, 20]] +
        [f'ts_close_to_vwap_z{w}' for w in [3, 5, 10, 20]] +
        [f'ts_turnover_per_vol_ma{w}' for w in [3, 5, 10, 20]] +
        [f'ts_turnover_per_vol_z{w}' for w in [3, 5, 10, 20]] +
        [f'ts_turnover_per_vol_rel{w}' for w in [3, 5, 10, 20]] +
        [f'ts_log_turnover_z{w}' for w in [3, 5, 10, 20]] +
        [f'ts_volume_z_ma{w}' for w in [3, 5, 10, 20]] +
        [f'ts_vwap_mom_ma{w}' for w in [3, 5, 10, 20]] +
        ['ts_turnover_per_vol_mom3', 'ts_turnover_per_vol_mom5']
    )

    extra_regime_feats = [
        'ret_5d', 'ret_10d', 'ret_20d',
        'range_mean_5', 'range_mean_10', 'range_mean_20',
        'range_z_20',
        'trend_to_vol_10', 'trend_to_vol_20',
        f'{benchmark_name}_trend_to_vol_10', f'{benchmark_name}_trend_to_vol_20',
        'alpha_to_vol_10', 'alpha_to_vol_20',
        'vol_ratio_5_20', f'{benchmark_name}_vol_ratio_5_20',
        'compression_10', 'compression_20'
    ]

    return {
        "base": base_feats,
        "context": context_feats,
        "flow": flow_feats,
        "regime": regime_feats,
        "extra_context": extra_context_feats,
        "extra_flow": extra_flow_feats,
        "extra_regime": extra_regime_feats,
    }

def build_feature_list(
    context_tickers,
    lag_steps,
    mom_windows,
    vol_windows,
    include=None,
    exclude=None,
    benchmark='imoex'
):
    groups = get_feature_groups(lag_steps, mom_windows, vol_windows, context_tickers, benchmark)

    if include is None:
        include = list(groups.keys())

    if exclude is None:
        exclude = []

    unknown = (set(include) | set(exclude)) - set(groups.keys())
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")

    features = []
    for name in include:
        if name not in exclude:
            features.extend(groups[name])

    return features

def _correlation(data, features, threshold, test_size, valid_size):
    train_size = len(data) - test_size - valid_size
    train = data[:train_size]
    
    corr_train = train[features].corr().abs()
    upper = corr_train.where(np.triu(np.ones(corr_train.shape), k=1).astype(bool))
    auto_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    features_final = [f for f in features if f not in auto_drop]
    return features_final