from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .backtest import backtest_probs, optimize_thresholds
from .dataset import make_target, train_test_valid_split, walk_forward_split
from .preprocessing import (
    DEFAULT_CONTEXT_TICKERS,
    DEFAULT_LAG_STEPS,
    DEFAULT_MOM_WINDOWS,
    DEFAULT_VOL_WINDOWS,
    preprocessing,
)
from .train_utils import (
    best_threshold,
    calibrate_probs,
    feature_importance,
    fast_catboost_random_search,
    train_catboost_classifier,
)


def _safe_metric(metric_fn, *args, **kwargs):
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return np.nan


def _predict_probs(model, X):
    probs = model.predict_proba(X)[:, 1]
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)


def _classification_metrics(y_true, probs, threshold, prefix):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    return {
        f"{prefix}_auc": _safe_metric(roc_auc_score, y_true, probs),
        f"{prefix}_acc": float(accuracy_score(y_true, preds)),
        f"{prefix}_brier": _safe_metric(brier_score_loss, y_true, probs),
        f"{prefix}_logloss": _safe_metric(log_loss, y_true, probs),
    }


def _normalize_horizons(horizons):
    if not horizons:
        raise ValueError("horizons must contain at least one horizon")

    normalized = []
    for horizon in horizons:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("Each horizon must be positive")
        if horizon not in normalized:
            normalized.append(horizon)

    return normalized


def _split_tail_frame(data, valid_size, test_size):
    if valid_size <= 0 or test_size <= 0:
        raise ValueError("valid_size and test_size must be positive")

    if len(data) < valid_size + test_size:
        raise ValueError(
            "Not enough rows to create validation and test splits. "
            f"Need at least {valid_size + test_size}, got {len(data)}"
        )

    split_valid = len(data) - valid_size - test_size
    split_test = len(data) - test_size

    return {
        "train": data.iloc[:split_valid].copy(),
        "valid": data.iloc[split_valid:split_test].copy(),
        "test": data.iloc[split_test:].copy(),
    }


def _resolve_decision_split_sizes(data_len, valid_size, test_size):
    valid_size = int(valid_size)
    test_size = int(test_size)

    if valid_size <= 0 or test_size <= 0:
        raise ValueError("decision_valid_size and decision_test_size must be positive")

    requested_valid_size = valid_size
    requested_test_size = test_size
    requested_total = requested_valid_size + requested_test_size

    if data_len < 2:
        raise ValueError(
            "Decision layer has too few rows after joining horizon predictions. "
            f"Available rows: {data_len}. Need at least 2 rows to create validation/test splits."
        )

    adjusted = False
    if requested_total > data_len:
        adjusted = True

        valid_ratio = requested_valid_size / requested_total
        valid_size = max(1, int(round(data_len * valid_ratio)))
        test_size = data_len - valid_size

        if test_size <= 0:
            test_size = 1
            valid_size = data_len - 1

        if valid_size <= 0:
            valid_size = 1
            test_size = data_len - 1

        if valid_size + test_size > data_len:
            test_size = data_len - valid_size

        if valid_size <= 0 or test_size <= 0:
            raise ValueError(
                "Decision layer could not auto-adjust validation/test sizes. "
                f"Available rows after joining predictions: {data_len}. "
                f"Requested valid/test: {requested_valid_size}/{requested_test_size}."
            )

    diagnostics = {
        "available_rows": int(data_len),
        "requested_valid_size": int(requested_valid_size),
        "requested_test_size": int(requested_test_size),
        "requested_total": int(requested_total),
        "used_valid_size": int(valid_size),
        "used_test_size": int(test_size),
        "used_total": int(valid_size + test_size),
        "auto_adjusted": bool(adjusted),
    }

    return valid_size, test_size, diagnostics


def _run_oos_walk_forward_for_horizon(
    data,
    features,
    *,
    horizon,
    valid_size,
    test_size,
    top_n_features,
    fold_train_size,
    fold_valid_size,
    fold_test_size,
    fold_step_size,
    iterations,
    early_stopping_rounds,
    random_state,
    fit_verbose,
    calibration_method,
    cost_bps,
    model_params=None,
    search_n_iter=25,
    search_n_splits=3,
):
    horizon_data = make_target(data=data.copy(), horizont=horizon)
    top_n_features = min(int(top_n_features), len(features))
    if top_n_features <= 0:
        raise ValueError("top_n_features must be positive")

    base_stage = _run_holdout_stage(
        data=horizon_data,
        features=features,
        valid_size=valid_size,
        test_size=test_size,
        horizon=horizon,
        model_params=model_params,
        search_n_iter=search_n_iter,
        search_n_splits=search_n_splits,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        fit_verbose=fit_verbose,
        calibration_method=calibration_method,
        cost_bps=cost_bps,
    )

    importance_df, importance_sum = feature_importance(
        base_stage["model"],
        features,
        n=top_n_features,
    )
    selected_features = importance_df["feature"].tolist()

    selected_stage = _run_holdout_stage(
        data=horizon_data,
        features=selected_features,
        valid_size=valid_size,
        test_size=test_size,
        horizon=horizon,
        model_params=base_stage["best_params"],
        search_n_iter=search_n_iter,
        search_n_splits=search_n_splits,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        fit_verbose=fit_verbose,
        calibration_method=calibration_method,
        cost_bps=cost_bps,
    )

    prediction_rows = []
    fold_rows = []

    try:
        for fold in walk_forward_split(
            data=horizon_data,
            features_final=selected_features,
            train_size=fold_train_size,
            valid_size=fold_valid_size,
            test_size=fold_test_size,
            step_size=fold_step_size,
            gap=horizon,
        ):
            model = train_catboost_classifier(
                X_train=fold.X_train,
                y_train=fold.y_train,
                X_valid=fold.X_valid,
                y_valid=fold.y_valid,
                params=base_stage["best_params"],
                iterations=iterations,
                eval_metric="Accuracy",
                random_state=random_state,
                verbose=fit_verbose,
                early_stopping_rounds=early_stopping_rounds,
            )

            valid_probs = _predict_probs(model, fold.X_valid)
            test_probs_raw = _predict_probs(model, fold.X_test)
            test_probs = calibrate_probs(
                valid_probs=valid_probs,
                y_valid=fold.y_valid,
                test_probs=test_probs_raw,
                method=calibration_method,
            )

            proba_col = f"proba_h{horizon}"
            score_col = f"score_h{horizon}"

            fold_pred = pd.DataFrame(
                {
                    "date": pd.to_datetime(fold.test["date"]).values,
                    proba_col: test_probs,
                    score_col: 2.0 * test_probs - 1.0,
                }
            )
            prediction_rows.append(fold_pred)

            fold_rows.append(
                {
                    "horizon": int(horizon),
                    "fold": int(fold.fold),
                    "fold_mode": "walk_forward",
                    "train_start": pd.to_datetime(fold.train["date"].iloc[0]),
                    "train_end": pd.to_datetime(fold.train["date"].iloc[-1]),
                    "valid_start": pd.to_datetime(fold.valid["date"].iloc[0]),
                    "valid_end": pd.to_datetime(fold.valid["date"].iloc[-1]),
                    "test_start": pd.to_datetime(fold.test["date"].iloc[0]),
                    "test_end": pd.to_datetime(fold.test["date"].iloc[-1]),
                    "best_iteration": int(model.get_best_iteration()),
                    "valid_auc": _safe_metric(roc_auc_score, fold.y_valid, valid_probs),
                    "test_auc_raw": _safe_metric(roc_auc_score, fold.y_test, test_probs_raw),
                    "test_auc_cal": _safe_metric(roc_auc_score, fold.y_test, test_probs),
                    "n_selected_features": int(len(selected_features)),
                }
            )
    except ValueError as exc:
        warnings.warn(
            f"Walk-forward validation for horizon={horizon} could not be created "
            f"with the current fold sizes. Falling back to the holdout test split. "
            f"Details: {exc}",
            stacklevel=2,
        )

        fallback_test = selected_stage["splits"]["test"]
        fallback_probs = np.asarray(selected_stage["splits"]["test_probs_cal"], dtype=float)
        proba_col = f"proba_h{horizon}"
        score_col = f"score_h{horizon}"

        prediction_rows = [
            pd.DataFrame(
                {
                    "date": pd.to_datetime(fallback_test["date"]).values,
                    proba_col: fallback_probs,
                    score_col: 2.0 * fallback_probs - 1.0,
                }
            )
        ]
        fold_rows = [
            {
                "horizon": int(horizon),
                "fold": -1,
                "fold_mode": "holdout_fallback",
                "train_start": pd.to_datetime(selected_stage["splits"]["train"]["date"].iloc[0]),
                "train_end": pd.to_datetime(selected_stage["splits"]["train"]["date"].iloc[-1]),
                "valid_start": pd.to_datetime(selected_stage["splits"]["valid"]["date"].iloc[0]),
                "valid_end": pd.to_datetime(selected_stage["splits"]["valid"]["date"].iloc[-1]),
                "test_start": pd.to_datetime(fallback_test["date"].iloc[0]),
                "test_end": pd.to_datetime(fallback_test["date"].iloc[-1]),
                "best_iteration": int(selected_stage["model"].get_best_iteration()),
                "valid_auc": float(selected_stage["metrics"]["valid_auc"]),
                "test_auc_raw": float(selected_stage["metrics"]["test_raw_auc"]),
                "test_auc_cal": float(selected_stage["metrics"]["test_cal_auc"]),
                "n_selected_features": int(len(selected_features)),
            }
        ]

    if not prediction_rows:
        raise ValueError(
            f"No walk-forward predictions were produced for horizon={horizon}. "
            "Check fold sizes and available history."
        )

    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    if predictions["date"].duplicated().any():
        dup_dates = (
            predictions.loc[predictions["date"].duplicated(), "date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        warnings.warn(
            f"Overlapping OOS predictions detected for horizon={horizon}. "
            f"Aggregating duplicate dates by mean probability/score. "
            f"Sample duplicate dates: {dup_dates[:5]}",
            stacklevel=2,
        )
        predictions = (
            predictions.groupby("date", as_index=False)
            .mean(numeric_only=True)
            .sort_values("date")
            .reset_index(drop=True)
        )

    return {
        "horizon": int(horizon),
        "best_params": base_stage["best_params"],
        "selected_features": selected_features,
        "feature_importance": importance_df.reset_index(drop=True),
        "feature_importance_sum": float(importance_sum),
        "base_stage": base_stage,
        "selected_stage": selected_stage,
        "predictions": predictions,
        "fold_metrics": pd.DataFrame(fold_rows),
    }


def _merge_horizon_predictions(horizon_results):
    merged = None

    for result in horizon_results:
        current = result["predictions"].copy()
        merged = current if merged is None else merged.merge(current, on="date", how="inner")

    if merged is None:
        return pd.DataFrame()

    if merged.empty:
        warnings.warn(
            "The common multi-horizon prediction frame is empty after inner join across horizons. "
            "Decision layer will be skipped, but per-horizon results are still available.",
            stacklevel=2,
        )
        return merged

    return merged.sort_values("date").reset_index(drop=True)


def _run_decision_layer(
    predictions,
    decision_data,
    *,
    score_cols,
    decision_valid_size,
    decision_test_size,
    decision_horizon,
    cost_bps,
    decision_score_col,
    decision_min_trades,
    decision_min_gap,
    decision_lower_grid,
    decision_upper_grid,
):
    if predictions.empty:
        return {
            "status": "skipped",
            "reason": "empty_common_prediction_frame",
            "data": pd.DataFrame(),
            "splits": {"train": pd.DataFrame(), "valid": pd.DataFrame(), "test": pd.DataFrame()},
            "split_diagnostics": {
                "available_rows": 0,
                "requested_valid_size": int(decision_valid_size),
                "requested_test_size": int(decision_test_size),
                "requested_total": int(decision_valid_size) + int(decision_test_size),
                "used_valid_size": 0,
                "used_test_size": 0,
                "used_total": 0,
                "auto_adjusted": False,
            },
            "long_threshold": np.nan,
            "short_threshold": np.nan,
            "threshold_metric": decision_score_col,
            "validation_threshold_search": None,
            "validation_backtest": None,
            "test_backtest": None,
        }

    frame = (
        predictions.merge(
            decision_data[["date", "fwd_ret", "y"]],
            on="date",
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    if frame.empty:
        return {
            "status": "skipped",
            "reason": "empty_join_with_decision_target",
            "data": frame,
            "splits": {"train": pd.DataFrame(), "valid": pd.DataFrame(), "test": pd.DataFrame()},
            "split_diagnostics": {
                "available_rows": 0,
                "requested_valid_size": int(decision_valid_size),
                "requested_test_size": int(decision_test_size),
                "requested_total": int(decision_valid_size) + int(decision_test_size),
                "used_valid_size": 0,
                "used_test_size": 0,
                "used_total": 0,
                "auto_adjusted": False,
            },
            "long_threshold": np.nan,
            "short_threshold": np.nan,
            "threshold_metric": decision_score_col,
            "validation_threshold_search": None,
            "validation_backtest": None,
            "test_backtest": None,
        }

    if len(frame) < 2:
        return {
            "status": "skipped",
            "reason": "too_few_rows_for_decision_split",
            "data": frame,
            "splits": {"train": pd.DataFrame(), "valid": pd.DataFrame(), "test": pd.DataFrame()},
            "split_diagnostics": {
                "available_rows": int(len(frame)),
                "requested_valid_size": int(decision_valid_size),
                "requested_test_size": int(decision_test_size),
                "requested_total": int(decision_valid_size) + int(decision_test_size),
                "used_valid_size": 0,
                "used_test_size": 0,
                "used_total": 0,
                "auto_adjusted": False,
            },
            "long_threshold": np.nan,
            "short_threshold": np.nan,
            "threshold_metric": decision_score_col,
            "validation_threshold_search": None,
            "validation_backtest": None,
            "test_backtest": None,
        }

    frame["mean_score"] = frame[score_cols].mean(axis=1)
    decision_valid_size, decision_test_size, split_diagnostics = _resolve_decision_split_sizes(
        len(frame),
        decision_valid_size,
        decision_test_size,
    )

    if split_diagnostics["auto_adjusted"]:
        warnings.warn(
            "Decision layer split sizes were auto-adjusted because the requested validation/test "
            f"window does not fit into the common multi-horizon prediction frame. "
            f"Available rows: {split_diagnostics['available_rows']}. "
            f"Requested valid/test: {split_diagnostics['requested_valid_size']}/"
            f"{split_diagnostics['requested_test_size']}. "
            f"Using valid/test: {split_diagnostics['used_valid_size']}/"
            f"{split_diagnostics['used_test_size']}.",
            stacklevel=2,
        )

    splits = _split_tail_frame(frame, decision_valid_size, decision_test_size)

    best_thresholds = optimize_thresholds(
        proba=splits["valid"]["mean_score"].values,
        fwd_ret=splits["valid"]["fwd_ret"].values,
        date=splits["valid"]["date"].values,
        cost_bps=cost_bps,
        ret_type="log",
        horizon=decision_horizon,
        min_trades=decision_min_trades,
        min_gap=decision_min_gap,
        score_col=decision_score_col,
        lower_grid=decision_lower_grid,
        upper_grid=decision_upper_grid,
    )
    if best_thresholds is None:
        warnings.warn(
            "No valid decision thresholds were found with the requested min_trades. "
            "Retrying threshold search with min_trades=0.",
            stacklevel=2,
        )
        best_thresholds = optimize_thresholds(
            proba=splits["valid"]["mean_score"].values,
            fwd_ret=splits["valid"]["fwd_ret"].values,
            date=splits["valid"]["date"].values,
            cost_bps=cost_bps,
            ret_type="log",
            horizon=decision_horizon,
            min_trades=0,
            min_gap=decision_min_gap,
            score_col=decision_score_col,
            lower_grid=decision_lower_grid,
            upper_grid=decision_upper_grid,
        )

    if best_thresholds is None:
        return {
            "status": "skipped",
            "reason": "threshold_search_failed",
            "data": frame,
            "splits": splits,
            "split_diagnostics": split_diagnostics,
            "long_threshold": np.nan,
            "short_threshold": np.nan,
            "threshold_metric": decision_score_col,
            "validation_threshold_search": None,
            "validation_backtest": None,
            "test_backtest": None,
        }

    valid_bt = backtest_probs(
        proba_long=splits["valid"]["mean_score"].values,
        fwd_ret=splits["valid"]["fwd_ret"].values,
        date=splits["valid"]["date"].values,
        lower=best_thresholds["lower"],
        upper=best_thresholds["upper"],
        cost_bps=cost_bps,
        ret_type="log",
        horizon=decision_horizon,
    )

    test_bt = backtest_probs(
        proba_long=splits["test"]["mean_score"].values,
        fwd_ret=splits["test"]["fwd_ret"].values,
        date=splits["test"]["date"].values,
        lower=best_thresholds["lower"],
        upper=best_thresholds["upper"],
        cost_bps=cost_bps,
        ret_type="log",
        horizon=decision_horizon,
    )

    return {
        "status": "ok",
        "reason": None,
        "data": frame,
        "splits": splits,
        "split_diagnostics": split_diagnostics,
        "long_threshold": float(best_thresholds["upper"]),
        "short_threshold": float(best_thresholds["lower"]),
        "threshold_metric": decision_score_col,
        "validation_threshold_search": best_thresholds,
        "validation_backtest": valid_bt,
        "test_backtest": test_bt,
    }


def _run_holdout_stage(
    data,
    features,
    *,
    valid_size,
    test_size,
    horizon=1,
    model_params=None,
    search_n_iter=25,
    search_n_splits=3,
    iterations=5000,
    early_stopping_rounds=200,
    random_state=42,
    fit_verbose=0,
    calibration_method="platt",
    cost_bps=10,
):
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        train,
        valid,
        test,
    ) = train_test_valid_split(
        data=data,
        features_final=features,
        test_size=test_size,
        valid_size=valid_size,
    )

    search_res = None
    best_params = model_params
    if best_params is None:
        search_res = fast_catboost_random_search(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            valid_fwd_ret=valid["fwd_ret"].values,
            valid_date=valid["date"].values,
            n_iter=search_n_iter,
            n_splits=search_n_splits,
            random_state=random_state,
            iterations=iterations,
            fit_verbose=fit_verbose,
            early_stopping_rounds=early_stopping_rounds,
            cost_bps=cost_bps,
            horizon=horizon,
        )
        best_params = search_res["best_params"]

    model = train_catboost_classifier(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        params=best_params,
        iterations=iterations,
        eval_metric="Accuracy",
        random_state=random_state,
        verbose=fit_verbose,
        early_stopping_rounds=early_stopping_rounds,
    )

    train_probs = _predict_probs(model, X_train)
    valid_probs = _predict_probs(model, X_valid)
    test_probs_raw = _predict_probs(model, X_test)
    test_probs_cal = calibrate_probs(
        valid_probs=valid_probs,
        y_valid=y_valid,
        test_probs=test_probs_raw,
        method=calibration_method,
    )

    threshold, threshold_sharpe = best_threshold(
        y_valid,
        valid_probs,
        valid["fwd_ret"].values,
        date=valid["date"].values,
        cost_bps=cost_bps,
        horizon=horizon,
    )

    backtest_res = backtest_probs(
        proba_long=test_probs_cal,
        fwd_ret=test["fwd_ret"].values,
        date=test["date"].values,
        lower=threshold - 1e-9,
        upper=threshold + 1e-9,
        cost_bps=cost_bps,
        ret_type="log",
        horizon=horizon,
    )

    metrics = {
        "n_features": int(len(features)),
        "best_threshold": float(threshold),
        "threshold_sharpe": float(threshold_sharpe),
        "best_iteration": int(model.get_best_iteration()),
        **_classification_metrics(y_train, train_probs, 0.5, "train"),
        **_classification_metrics(y_valid, valid_probs, threshold, "valid"),
        **_classification_metrics(y_test, test_probs_raw, threshold, "test_raw"),
        **_classification_metrics(y_test, test_probs_cal, threshold, "test_cal"),
        "bt_total_return": float(backtest_res["metrics"]["total_return"]),
        "bt_cagr": float(backtest_res["metrics"]["cagr"]),
        "bt_sharpe": float(backtest_res["metrics"]["sharpe"]),
        "bt_sortino": float(backtest_res["metrics"]["sortino"]),
        "bt_max_drawdown": float(backtest_res["metrics"]["max_drawdown"]),
        "bt_active_hit_rate": float(backtest_res["metrics"]["active_hit_rate"]),
        "bt_trades": int(backtest_res["metrics"]["trades"]),
    }

    return {
        "model": model,
        "best_params": best_params,
        "search_results": search_res,
        "metrics": metrics,
        "splits": {
            "train": train,
            "valid": valid,
            "test": test,
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
            "train_probs": train_probs,
            "valid_probs": valid_probs,
            "test_probs_raw": test_probs_raw,
            "test_probs_cal": test_probs_cal,
        },
        "backtest": backtest_res,
    }


def _run_fold_validation(
    data,
    features,
    model_params,
    *,
    horizon,
    fold_train_size,
    fold_valid_size,
    fold_test_size,
    fold_step_size,
    fold_gap,
    iterations,
    early_stopping_rounds,
    random_state,
    fit_verbose,
    calibration_method,
    cost_bps,
):
    rows = []
    try:
        for fold in walk_forward_split(
            data=data,
            features_final=features,
            train_size=fold_train_size,
            valid_size=fold_valid_size,
            test_size=fold_test_size,
            step_size=fold_step_size,
            gap=fold_gap,
        ):
            model = train_catboost_classifier(
                X_train=fold.X_train,
                y_train=fold.y_train,
                X_valid=fold.X_valid,
                y_valid=fold.y_valid,
                params=model_params,
                iterations=iterations,
                eval_metric="Accuracy",
                random_state=random_state,
                verbose=fit_verbose,
                early_stopping_rounds=early_stopping_rounds,
            )

            valid_probs = _predict_probs(model, fold.X_valid)
            test_probs_raw = _predict_probs(model, fold.X_test)
            test_probs_cal = calibrate_probs(
                valid_probs=valid_probs,
                y_valid=fold.y_valid,
                test_probs=test_probs_raw,
                method=calibration_method,
            )

            threshold, threshold_sharpe = best_threshold(
                fold.y_valid,
                valid_probs,
                fold.valid["fwd_ret"].values,
                date=fold.valid["date"].values,
                cost_bps=cost_bps,
                horizon=horizon,
            )

            backtest_res = backtest_probs(
                proba_long=test_probs_cal,
                fwd_ret=fold.test["fwd_ret"].values,
                date=fold.test["date"].values,
                lower=threshold - 1e-9,
                upper=threshold + 1e-9,
                cost_bps=cost_bps,
                ret_type="log",
                horizon=horizon,
            )

            row = {
                "fold": int(fold.fold),
                "train_start": pd.to_datetime(fold.train["date"].iloc[0]),
                "train_end": pd.to_datetime(fold.train["date"].iloc[-1]),
                "valid_start": pd.to_datetime(fold.valid["date"].iloc[0]),
                "valid_end": pd.to_datetime(fold.valid["date"].iloc[-1]),
                "test_start": pd.to_datetime(fold.test["date"].iloc[0]),
                "test_end": pd.to_datetime(fold.test["date"].iloc[-1]),
                "n_features": int(len(features)),
                "best_threshold": float(threshold),
                "threshold_sharpe": float(threshold_sharpe),
                "best_iteration": int(model.get_best_iteration()),
                **_classification_metrics(fold.y_valid, valid_probs, threshold, "valid"),
                **_classification_metrics(fold.y_test, test_probs_raw, threshold, "test_raw"),
                **_classification_metrics(fold.y_test, test_probs_cal, threshold, "test_cal"),
                "bt_total_return": float(backtest_res["metrics"]["total_return"]),
                "bt_cagr": float(backtest_res["metrics"]["cagr"]),
                "bt_sharpe": float(backtest_res["metrics"]["sharpe"]),
                "bt_sortino": float(backtest_res["metrics"]["sortino"]),
                "bt_max_drawdown": float(backtest_res["metrics"]["max_drawdown"]),
                "bt_active_hit_rate": float(backtest_res["metrics"]["active_hit_rate"]),
                "bt_trades": int(backtest_res["metrics"]["trades"]),
            }
            rows.append(row)
    except ValueError as exc:
        warnings.warn(
            "Walk-forward validation could not be created with the current fold sizes. "
            f"Returning an empty fold metrics table instead. Details: {exc}",
            stacklevel=2,
        )
        return pd.DataFrame()

    return pd.DataFrame(rows)


def run_catboost_feature_selection_pipeline(
    data,
    features_final,
    *,
    valid_size,
    test_size,
    top_n_features=30,
    horizont=None,
    search_n_iter=25,
    search_n_splits=3,
    iterations=5000,
    early_stopping_rounds=200,
    random_state=42,
    fit_verbose=0,
    base_model_params=None,
    selected_model_params=None,
    fold_train_size=252,
    fold_valid_size=252 // 3,
    fold_test_size=252 // 6,
    fold_step_size=21 * 3,
    calibration_method="platt",
    cost_bps=10,
):
    data = data.copy().sort_values("date").reset_index(drop=True)
    if horizont is not None:
        horizont = int(horizont)
        if horizont <= 0:
            raise ValueError("horizont must be positive")
        data = make_target(data=data, horizont=horizont)
        fold_gap = horizont
    else:
        fold_gap = 1

    required_cols = {"date", "fwd_ret", "y"}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(
            "Data must contain target columns. "
            f"Missing columns: {sorted(missing_cols)}"
        )

    if not features_final:
        raise ValueError("features_final must contain at least one feature")

    top_n_features = min(int(top_n_features), len(features_final))
    if top_n_features <= 0:
        raise ValueError("top_n_features must be positive")

    base_stage = _run_holdout_stage(
        data=data,
        features=features_final,
        valid_size=valid_size,
        test_size=test_size,
        horizon=horizont or 1,
        model_params=base_model_params,
        search_n_iter=search_n_iter,
        search_n_splits=search_n_splits,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        fit_verbose=fit_verbose,
        calibration_method=calibration_method,
        cost_bps=cost_bps,
    )

    importance_df, importance_sum = feature_importance(
        base_stage["model"],
        features_final,
        n=top_n_features,
    )
    selected_features = importance_df["feature"].tolist()

    selected_stage = _run_holdout_stage(
        data=data,
        features=selected_features,
        valid_size=valid_size,
        test_size=test_size,
        horizon=horizont or 1,
        model_params=selected_model_params,
        search_n_iter=search_n_iter,
        search_n_splits=search_n_splits,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        fit_verbose=fit_verbose,
        calibration_method=calibration_method,
        cost_bps=cost_bps,
    )

    fold_metrics = _run_fold_validation(
        data=data,
        features=selected_features,
        model_params=selected_stage["best_params"],
        horizon=horizont or 1,
        fold_train_size=fold_train_size,
        fold_valid_size=fold_valid_size,
        fold_test_size=fold_test_size,
        fold_step_size=fold_step_size,
        fold_gap=fold_gap,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        fit_verbose=fit_verbose,
        calibration_method=calibration_method,
        cost_bps=cost_bps,
    )

    fold_summary = (
        fold_metrics.select_dtypes(include=[np.number]).agg(["mean", "std", "min", "max"]).T
        if not fold_metrics.empty
        else pd.DataFrame()
    )

    return {
        "data": data,
        "all_features": list(features_final),
        "selected_features": selected_features,
        "feature_importance": importance_df.reset_index(drop=True),
        "feature_importance_sum": float(importance_sum),
        "base_stage": base_stage,
        "selected_stage": selected_stage,
        "fold_metrics": fold_metrics,
        "fold_summary": fold_summary,
    }


def multi_horizon_catboost_pipeline(
    *,
    ticker,
    start_date,
    end_date,
    horizons,
    period="1d",
    valid_size=252,
    test_size=252 // 3,
    threshold=0.995,
    top_n_features=30,
    search_n_iter=25,
    search_n_splits=3,
    iterations=5000,
    early_stopping_rounds=200,
    random_state=42,
    fit_verbose=0,
    model_params=None,
    model_params_by_horizon=None,
    fold_train_size=252,
    fold_valid_size=252 // 3,
    fold_test_size=252 // 6,
    fold_step_size=21 * 3,
    calibration_method="platt",
    cost_bps=10,
    decision_horizon=None,
    decision_valid_size=None,
    decision_test_size=None,
    decision_score_col="sharpe",
    decision_min_trades=10,
    decision_min_gap=0.1,
    decision_lower_grid=None,
    decision_upper_grid=None,
    lag_steps: Sequence[int] = DEFAULT_LAG_STEPS,
    mom_windows: Sequence[int] = DEFAULT_MOM_WINDOWS,
    vol_windows: Sequence[int] = DEFAULT_VOL_WINDOWS,
    context_tickers: Sequence[str] = DEFAULT_CONTEXT_TICKERS,
    benchmark="imoex",
) -> dict[str, Any]:
    horizons = _normalize_horizons(horizons)

    if decision_horizon is None:
        decision_horizon = min(horizons)
    decision_horizon = int(decision_horizon)
    if decision_horizon <= 0:
        raise ValueError("decision_horizon must be positive")

    if decision_valid_size is None:
        decision_valid_size = valid_size
    if decision_test_size is None:
        decision_test_size = test_size

    if decision_lower_grid is None:
        decision_lower_grid = np.linspace(-0.95, -0.05, 19)
    if decision_upper_grid is None:
        decision_upper_grid = np.linspace(0.05, 0.95, 19)

    base_data, features_final = preprocessing(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        period=period,
        valid_size=valid_size,
        test_size=test_size,
        threshold=threshold,
        lag_steps=lag_steps,
        mom_windows=mom_windows,
        vol_windows=vol_windows,
        context_tickers=context_tickers,
        benchmark=benchmark,
    )

    horizon_results = []
    for horizon in horizons:
        cur_params = model_params
        if model_params_by_horizon is not None:
            cur_params = model_params_by_horizon.get(horizon, cur_params)

        horizon_results.append(
            _run_oos_walk_forward_for_horizon(
                data=base_data,
                features=features_final,
                horizon=horizon,
                valid_size=valid_size,
                test_size=test_size,
                top_n_features=top_n_features,
                fold_train_size=fold_train_size,
                fold_valid_size=fold_valid_size,
                fold_test_size=fold_test_size,
                fold_step_size=fold_step_size,
                iterations=iterations,
                early_stopping_rounds=early_stopping_rounds,
                random_state=random_state,
                fit_verbose=fit_verbose,
                calibration_method=calibration_method,
                cost_bps=cost_bps,
                model_params=cur_params,
                search_n_iter=search_n_iter,
                search_n_splits=search_n_splits,
            )
        )

    predictions = _merge_horizon_predictions(horizon_results)

    proba_cols = [f"proba_h{h}" for h in horizons]
    score_cols = [f"score_h{h}" for h in horizons]
    predictions = predictions[["date", *proba_cols, *score_cols]].copy()

    decision_data = make_target(base_data.copy(), horizont=decision_horizon)
    decision_layer = _run_decision_layer(
        predictions=predictions,
        decision_data=decision_data,
        score_cols=score_cols,
        decision_valid_size=decision_valid_size,
        decision_test_size=decision_test_size,
        decision_horizon=decision_horizon,
        cost_bps=cost_bps,
        decision_score_col=decision_score_col,
        decision_min_trades=decision_min_trades,
        decision_min_gap=decision_min_gap,
        decision_lower_grid=decision_lower_grid,
        decision_upper_grid=decision_upper_grid,
    )

    horizon_fold_metrics = pd.concat(
        [result["fold_metrics"] for result in horizon_results],
        ignore_index=True,
    )

    return {
        "data": base_data,
        "features": list(features_final),
        "horizons": horizons,
        "n_models": len(horizons),
        "per_horizon": {result["horizon"]: result for result in horizon_results},
        "predictions": predictions,
        "decision_layer": decision_layer,
        "fold_metrics": horizon_fold_metrics,
    }


def catboost_learning_pipeline(
    *,
    ticker,
    start_date,
    end_date,
    period="1d",
    valid_size=252,
    test_size=252 // 3,
    threshold=0.995,
    horizont=1,
    top_n_features=30,
    search_n_iter=25,
    search_n_splits=3,
    iterations=5000,
    early_stopping_rounds=200,
    random_state=42,
    fit_verbose=0,
    base_model_params=None,
    selected_model_params=None,
    fold_train_size=252,
    fold_valid_size=252 // 3,
    fold_test_size=252 // 6,
    fold_step_size=21 * 3,
    calibration_method="platt",
    cost_bps=10,
    lag_steps: Sequence[int] = DEFAULT_LAG_STEPS,
    mom_windows: Sequence[int] = DEFAULT_MOM_WINDOWS,
    vol_windows: Sequence[int] = DEFAULT_VOL_WINDOWS,
    context_tickers: Sequence[str] = DEFAULT_CONTEXT_TICKERS,
    benchmark="imoex",
) -> dict[str, Any]:
    horizont = int(horizont)
    if horizont <= 0:
        raise ValueError("horizont must be positive")

    data, features_final = preprocessing(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        period=period,
        valid_size=valid_size,
        test_size=test_size,
        threshold=threshold,
        lag_steps=lag_steps,
        mom_windows=mom_windows,
        vol_windows=vol_windows,
        context_tickers=context_tickers,
        benchmark=benchmark,
    )

    return run_catboost_feature_selection_pipeline(
        data=data,
        features_final=features_final,
        valid_size=valid_size,
        test_size=test_size,
        top_n_features=top_n_features,
        horizont=horizont,
        search_n_iter=search_n_iter,
        search_n_splits=search_n_splits,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        fit_verbose=fit_verbose,
        base_model_params=base_model_params,
        selected_model_params=selected_model_params,
        fold_train_size=fold_train_size,
        fold_valid_size=fold_valid_size,
        fold_test_size=fold_test_size,
        fold_step_size=fold_step_size,
        calibration_method=calibration_method,
        cost_bps=cost_bps,
    )


catboost_learning = catboost_learning_pipeline
multi_horizon_pipeline = multi_horizon_catboost_pipeline
