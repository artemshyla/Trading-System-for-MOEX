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

def eval(model, X, y, thr=0.5):
  probs = model.predict_proba(X)[:,1]
  preds = (probs >= thr).astype(int)
  acc = accuracy_score(y, preds)
  auc = roc_auc_score(y, probs)
  return auc, acc, probs

def best_threshold(y_true, probs):
  thrs = np.linspace(0.1, 0.9, 100)
  best_thr, best_acc = 0.5, -1
  for thr in thrs:
    preds = (probs >= thr).astype(int)
    acc = accuracy_score(y_true, preds)
    if acc > best_acc:
      best_acc = acc
      best_thr = thr
  return best_thr, best_acc

def fearure_importance(catboost_model, features_final, n=10):
    cat_importance = pd.DataFrame({
        'feature': features_final,
        'importance': catboost_model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    return cat_importance.head(n), sum(cat_importance['importance'].head(n))