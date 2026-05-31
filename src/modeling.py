import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.features import ensure_stationarity, build_feature_matrix, filter_multicollinearity


def fit_ols(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, pd.Series]:
    X_mat = X.to_numpy()
    y_mat = y.to_numpy()
    beta, _, _, _ = np.linalg.lstsq(X_mat, y_mat, rcond=None)
    residuals = y - X @ beta
    return beta, residuals


def evaluate_residuals(residuals: pd.Series) -> dict:
    dw_stat = durbin_watson(residuals)
    n_obs = len(residuals)
    lags = min(12, max(1, n_obs // 5))
    try:
        lb_df = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        ljung_box_p = float(lb_df["lb_pvalue"].iloc[0])
    except Exception:
        ljung_box_p = 1.0

    return {
        "dw_stat": float(dw_stat),
        "ljung_box_p": ljung_box_p
    }


def fit_adaptive_ols(
        y_stat: pd.Series,
        exog_series: pd.Series,
        base_lags: int,
        max_increment: int = 4,
        include_trend: bool = False,
        include_exog: bool = False,
        seasonal_style: str = "Fourier & Peak Dummies"
) -> tuple[np.ndarray, list[str], int, pd.Series, dict]:
    best_beta = None
    best_features = None
    chosen_lags = base_lags
    residuals = None
    diagnostics = {}

    for lag in range(base_lags, base_lags + max_increment + 1):
        X, y_aligned = build_feature_matrix(
            y_stat,
            exog_series,
            max_lags=lag,
            include_trend=include_trend,
            include_exog=include_exog,
            seasonal_style=seasonal_style
        )
        X_filtered = filter_multicollinearity(X)
        beta, res = fit_ols(X_filtered, y_aligned)
        diag = evaluate_residuals(res)

        best_beta = beta
        best_features = list(X_filtered.columns)
        chosen_lags = lag
        residuals = res
        diagnostics = diag

        if (1.5 <= diag["dw_stat"] <= 2.5) and (diag["ljung_box_p"] > 0.05):
            break

    return best_beta, best_features, chosen_lags, residuals, diagnostics


def generate_dynamic_forecast(
        history_z: pd.Series,
        history_w: pd.Series,
        exog_all: pd.Series,
        beta: np.ndarray,
        feature_cols: list[str],
        max_lags: int,
        d: int,
        horizon: int,
        trend_offset: int,
        include_trend: bool = False,
        include_exog: bool = False,
        seasonal_style: str = "Fourier & Peak Dummies"
) -> tuple[np.ndarray, np.ndarray]:
    pred_z = []
    pred_w = []

    curr_z = list(history_z.to_numpy())
    curr_w = list(history_w.to_numpy())

    is_non_negative = np.min(curr_w) >= 0.0

    for h in range(1, horizon + 1):
        row = {"const": 1.0}

        for lag in range(1, max_lags + 1):
            row[f"{history_z.name}_lag_{lag}"] = curr_z[-lag]

        target_idx = len(history_z) + d + h - 1
        target_date = exog_all.index[target_idx] if target_idx < len(exog_all) else exog_all.index[-1]

        if include_exog:
            if target_idx < len(exog_all):
                row["exog_USD"] = exog_all.iloc[target_idx]
            else:
                row["exog_USD"] = exog_all.iloc[-1]

        if include_trend:
            row["trend"] = len(history_z) + h - 1

        month_val = target_date.month
        if seasonal_style == "Fourier & Peak Dummies":
            row["sin_month"] = np.sin(2 * np.pi * month_val / 12.0)
            row["cos_month"] = np.cos(2 * np.pi * month_val / 12.0)
            row["Dummy_Nov"] = 1.0 if month_val == 11 else 0.0
            row["Dummy_Dec"] = 1.0 if month_val == 12 else 0.0
        elif seasonal_style == "Full Monthly Dummies":
            for m in range(2, 13):
                row[f"month_{m}"] = 1.0 if month_val == m else 0.0

        feat_vector = np.array([row.get(col, 0.0) for col in feature_cols])

        z_next = float(feat_vector @ beta)
        pred_z.append(z_next)
        curr_z.append(z_next)

        if d == 1:
            w_next = curr_w[-1] + z_next
        elif d == 2:
            w_next = 2 * curr_w[-1] - curr_w[-2] + z_next
        else:
            w_next = z_next

        if is_non_negative:
            w_next = max(0.0, w_next)

        pred_w.append(w_next)
        curr_w.append(w_next)

    return np.array(pred_z), np.array(pred_w)


def rolling_window_backtest(
    df: pd.DataFrame,
    target_col: str,
    exog_col: str,
    train_window: int,
    horizon: int,
    max_lags: int = 4,
    include_trend: bool = False,
    include_exog: bool = False,
    seasonal_style: str = "Fourier & Peak Dummies",
    selected_anomalies: list = None,
    ema_alpha: float = 0.3,
) -> dict:
    n = len(df)
    results = []

    for i in range(0, n - train_window - horizon + 1, horizon):
        train_df = df.iloc[i : train_window + i]
        test_df = df.iloc[train_window + i : train_window + i + horizon]

        y_raw = train_df[target_col]
        y_active = y_raw.copy()

        if selected_anomalies:
            ema = y_raw.ewm(alpha=ema_alpha, adjust=False).mean()
            for date_str in selected_anomalies:
                target_date = pd.to_datetime(date_str)
                if target_date in y_active.index:
                    y_active.loc[target_date] = ema.loc[target_date]

        y_stat, d = ensure_stationarity(y_active, max_diff=1)

        beta, features, chosen_lags, _, _ = fit_adaptive_ols(
            y_stat=y_stat,
            exog_series=train_df[exog_col],
            base_lags=max_lags,
            include_trend=include_trend,
            include_exog=include_exog,
            seasonal_style=seasonal_style,
        )

        # Calculates where the stationarity transformation cropped the starting indices
        trend_start = len(train_df) - len(y_stat)
        exog_combined = pd.concat([train_df[exog_col], test_df[exog_col]])
        y_train_slice = y_active.loc[y_stat.index]

        _, pred_w = generate_dynamic_forecast(
            history_z=y_stat,
            history_w=y_train_slice,
            exog_all=exog_combined,
            beta=beta,
            feature_cols=features,
            max_lags=chosen_lags,
            d=d,
            horizon=horizon,
            trend_offset=trend_start,
            include_trend=include_trend,
            include_exog=include_exog,
            seasonal_style=seasonal_style,
        )

        pred_raw = pd.Series(pred_w, index=test_df.index)

        results.append(
            pd.DataFrame(
                {
                    "actual": test_df[target_col],
                    "forecast": pred_raw,
                },
                index=test_df.index,
            )
        )

    # Combine dataframes to natively handle overlapping dates or structural gaps
    eval_df = pd.concat(results).groupby(level=0).mean()

    all_actuals = eval_df["actual"].to_numpy()
    all_forecasts = eval_df["forecast"].to_numpy()

    mae = np.mean(np.abs(all_actuals - all_forecasts))
    rmse = np.sqrt(np.mean((all_actuals - all_forecasts) ** 2))

    denominator = np.abs(all_actuals) + np.abs(all_forecasts)
    smape = (
        np.mean(
            np.where(
                denominator < 1e-8,
                0.0,
                2.0 * np.abs(all_actuals - all_forecasts) / denominator,
            )
        )
        * 100
    )

    sum_actuals = np.sum(np.abs(all_actuals))
    wmape = (
        (np.sum(np.abs(all_actuals - all_forecasts)) / sum_actuals * 100)
        if sum_actuals > 1e-8
        else 0.0
    )

    return {
        "dates": eval_df.index.tolist(),
        "actuals": all_actuals.tolist(),
        "forecasts": all_forecasts.tolist(),
        "metrics": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "SMAPE": float(smape),
            "WMAPE": float(wmape),
        },
    }