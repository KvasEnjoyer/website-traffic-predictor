import numpy as np
import pandas as pd


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m")
    df.set_index("Time", inplace=True)
    df = df.ffill().bfill()
    return df.astype(np.float64)


def detect_and_smooth_anomalies(
    series: pd.Series,
    window: int = 12,
    threshold: float = 3.0,
    alpha: float = 0.3
) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0.0)

    std_safe = np.where(rolling_std == 0.0, 1e-9, rolling_std)
    z_scores = (series - rolling_mean) / std_safe

    anomalies = np.abs(z_scores) > threshold

    ema = series.ewm(alpha=alpha, adjust=False).mean()
    cleaned = series.copy()
    cleaned[anomalies] = ema[anomalies]

    return cleaned