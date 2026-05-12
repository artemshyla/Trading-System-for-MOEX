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
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

import moexalgo
from moexalgo import Ticker
import moexalgo.engines.currency
from moexalgo import Market

from .backtest import backtest_probs


def build_catboost_classifier(
    params=None,
    *,
    iterations=5000,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_state=42,
    verbose=0,
    thread_count=-1,
):
    model_params = {
        "iterations": iterations,
        "loss_function": loss_function,
        "eval_metric": eval_metric,
        "random_seed": random_state,
        "verbose": verbose,
        "auto_class_weights": "Balanced",
        "bootstrap_type": "Bayesian",
        "has_time": True,
        "thread_count": thread_count,
    }

    if params:
        model_params.update(params)

    return CatBoostClassifier(**model_params)


def train_catboost_classifier(
    X_train,
    y_train,
    X_valid,
    y_valid,
    params=None,
    *,
    iterations=5000,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_state=42,
    verbose=0,
    early_stopping_rounds=200,
    thread_count=-1,
):
    model = build_catboost_classifier(
        params=params,
        iterations=iterations,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_state=random_state,
        verbose=verbose,
        thread_count=thread_count,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
    )

    return model

def eval(model, X, y, thr=0.5):
  probs = model.predict_proba(X)[:,1]
  preds = (probs >= thr).astype(int)
  acc = accuracy_score(y, preds)
  auc = roc_auc_score(y, probs)
  return auc, acc, probs

def best_threshold(
    y_true,
    probs,
    fwd_ret,
    date=None,
    cost_bps=10,
    ret_type="log",
    horizon=1,
):
  probs = np.asarray(probs, dtype=float)
  fwd_ret = np.asarray(fwd_ret, dtype=float)

  if len(probs) != len(fwd_ret):
    raise ValueError("len(probs) must be equal to len(fwd_ret)")

  if y_true is not None and len(y_true) != len(probs):
    raise ValueError("len(y_true) must be equal to len(probs)")

  thrs = np.linspace(0.1, 0.9, 100)
  best_thr, best_sharpe = 0.5, -np.inf

  for thr in thrs:
    bt_res = backtest_probs(
        proba_long=probs,
        fwd_ret=fwd_ret,
        date=date,
        lower=thr - 1e-9,
        upper=thr + 1e-9,
        cost_bps=cost_bps,
        ret_type=ret_type,
        horizon=horizon,
    )
    sharpe = float(bt_res["metrics"]["sharpe"])

    if sharpe > best_sharpe:
      best_sharpe = sharpe
      best_thr = thr

  return best_thr, best_sharpe

def feature_importance(catboost_model, features_final, n=10):
    cat_importance = pd.DataFrame({
        'feature': features_final,
        'importance': catboost_model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    return cat_importance.head(n), sum(cat_importance['importance'].head(n))

def calibrate_probs(valid_probs, y_valid, test_probs, method='platt'):
    valid_probs = np.asarray(valid_probs, dtype=float)
    test_probs = np.asarray(test_probs, dtype=float)
    y_valid = np.asarray(y_valid, dtype=int)

    if len(np.unique(y_valid)) < 2:
        p = float(y_valid.mean())
        return np.full(len(test_probs), p, dtype=float)

    if method == 'platt':
        calibrator = LogisticRegression()
        calibrator.fit(valid_probs.reshape(-1, 1), y_valid)
        cal_probs = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    else:
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(valid_probs, y_valid)
        cal_probs = calibrator.predict(test_probs)

    return np.clip(cal_probs, 1e-6, 1 - 1e-6)

def fast_catboost_random_search(
    X_train,
    y_train,
    X_valid,
    y_valid,
    valid_fwd_ret,
    valid_date=None,
    n_iter=25,
    n_splits=3,
    random_state=42,
    iterations=5000,
    fit_verbose=200,
    early_stopping_rounds=200,
    cost_bps=10,
    horizon=1,
):
    param_distributions = {
        "depth": randint(3, 9),
        "learning_rate": uniform(0.01, 0.09),
        "l2_leaf_reg": uniform(1.0, 9.0),
        "min_data_in_leaf": randint(5, 31),
        "random_strength": uniform(0.0, 2.0),
        "bagging_temperature": uniform(0.0, 2.0),
        "border_count": [32, 64, 128, 254],
        "rsm": uniform(0.6, 0.4),
    }

    base_model = CatBoostClassifier(
        iterations=500,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_state,
        verbose=0,
        auto_class_weights="Balanced",
        bootstrap_type="Bayesian",
        has_time=True,
        thread_count=-1
    )

    cv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        verbose=1,
        n_jobs=1,
        refit=True,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_

    best_model = build_catboost_classifier(
        params=best_params,
        iterations=iterations,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_state=random_state,
        verbose=fit_verbose,
    )

    best_model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds
    )

    valid_auc, valid_acc, valid_probs = eval(best_model, X_valid, y_valid)
    threshold, threshold_sharpe = best_threshold(
        y_valid,
        valid_probs,
        valid_fwd_ret,
        date=valid_date,
        cost_bps=cost_bps,
        horizon=horizon,
    )

    search_results = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )

    cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "param_depth",
        "param_learning_rate",
        "param_l2_leaf_reg",
        "param_min_data_in_leaf",
        "param_random_strength",
        "param_bagging_temperature",
        "param_border_count",
        "param_rsm",
    ]
    cols = [col for col in cols if col in search_results.columns]
    search_results = search_results[cols]

    return {
        "search": search,
        "best_model": best_model,
        "best_params": best_params,
        "best_iteration": best_model.get_best_iteration(),
        "valid_auc": valid_auc,
        "valid_acc": valid_acc,
        "best_threshold": threshold,
        "threshold_sharpe": threshold_sharpe,
        "search_results": search_results
    }
