import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor


def ensure_stationarity(series: pd.Series, max_diff: int = 1) -> tuple[pd.Series, int]:
    current_series = series.copy()
    for d in range(max_diff + 1):
        cleaned_series = current_series.dropna()
        if len(cleaned_series) < 10:
            return current_series, d

        try:
            result = adfuller(cleaned_series)
            p_value = result[1]
            if p_value < 0.05:
                return current_series, d
        except Exception:
            pass

        if d < max_diff:
            current_series = current_series.diff()

    return current_series, max_diff


def build_feature_matrix(
        target_series: pd.Series,
        exog_series: pd.Series,
        max_lags: int,
        include_trend: bool = False,
        include_exog: bool = False,
        seasonal_style: str = "Fourier & Peak Dummies"
) -> tuple[pd.DataFrame, pd.Series]:
    df_features = pd.DataFrame(index=target_series.index)

    # Авторегресівні лаги
    for lag in range(1, max_lags + 1):
        df_features[f"{target_series.name}_lag_{lag}"] = target_series.shift(lag)

    # Екзогенні змінні
    if include_exog and exog_series is not None:
        df_features["exog_USD"] = exog_series

    # Л. тренд
    if include_trend:
        df_features["trend"] = np.arange(len(target_series))

    months = target_series.index.month
    if seasonal_style == "Fourier & Peak Dummies":
        df_features["sin_month"] = np.sin(2 * np.pi * months / 12.0)
        df_features["cos_month"] = np.cos(2 * np.pi * months / 12.0)
        df_features["Dummy_Nov"] = (months == 11).astype(float)
        df_features["Dummy_Dec"] = (months == 12).astype(float)
    elif seasonal_style == "Full Monthly Dummies":
        for m in range(2, 13):
            df_features[f"month_{m}"] = (months == m).astype(float)

    df_features["const"] = 1.0

    combined = pd.concat([target_series, df_features], axis=1).dropna()
    y_aligned = combined.iloc[:, 0]
    X_aligned = combined.iloc[:, 1:].astype(np.float64)

    return X_aligned, y_aligned


def filter_multicollinearity(X: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    cols_to_check = [col for col in X.columns if col != "const"]

    while len(cols_to_check) > 1:
        vifs = []
        for col in cols_to_check:
            temp_df = X[["const"] + cols_to_check]
            idx = temp_df.columns.get_loc(col)
            try:
                vif = variance_inflation_factor(temp_df.to_numpy(), idx)
                if np.isnan(vif) or np.isinf(vif):
                    vif = 1e10
            except Exception:
                vif = 1e10
            vifs.append(vif)

        max_vif = max(vifs)
        if max_vif > threshold:
            max_idx = vifs.index(max_vif)
            cols_to_check.pop(max_idx)
        else:
            break

    return X[["const"] + cols_to_check]