from __future__ import annotations

import os
import copy
import random
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Iterator

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

def make_target(data, horizont):
    data = data.sort_values("date").reset_index(drop=True)
    data["fwd_ret"] = np.log(data["close"].shift(-horizont) / data["close"])
    data["y"] = (data["fwd_ret"] > 0).astype(int)
    data = data.sort_values("date").reset_index(drop=True)
    
    data = data.dropna().reset_index(drop=True)  

    return data

def train_test_valid_split(data, features_final, test_size, valid_size):
    train_size = len(data) - test_size - valid_size
    train = data[:train_size]
    valid = data[train_size: (train_size + valid_size)]
    test = data[(train_size + valid_size): ]

    X_train = train[features_final]
    y_train = train["y"]
    
    X_valid = valid[features_final]
    y_valid = valid["y"]
    
    X_test = test[features_final]
    y_test = test["y"]
    return X_train, y_train, X_valid, y_valid, X_test, y_test, train, valid, test

@dataclass
class Fold:
    fold: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train: pd.DataFrame
    valid: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series

def walk_forward_split(
    data,
    features_final,
    train_size,
    valid_size,
    step_size=21,
    gap=1
) -> Iterator[Fold]:

    data = data.copy().sort_values('date').reset_index(drop=True)

    n = len(data)
    min_required = train_size + gap + valid_size
    if n < min_required:
        raise ValueError(f'Недостаточно строк. Нужно минимум {min_required}')

    fold = 0
    train_start = 0 
    train_end = train_size

    while train_end + gap + valid_size <= n:
        valid_start = train_end + gap
        valid_end = valid_start + valid_size

        train_idx = np.arange(train_start, train_end)
        valid_idx = np.arange(valid_start, valid_end)

        train = data.iloc[train_idx].copy()
        valid = data.iloc[valid_idx].copy()

        X_train = train[list(features_final)].copy()
        y_train = train['y'].copy()
        
        X_valid = valid[list(features_final)].copy()
        y_valid = valid['y'].copy()

        yield Fold(
            fold=fold,
            train_idx=train_idx,
            valid_idx=valid_idx,
            train=train,
            valid=valid,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid
        )

        fold += 1

        train_start += step_size
        train_end += step_size