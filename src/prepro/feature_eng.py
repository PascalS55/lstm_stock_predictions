import pandas as pd
import numpy as np


def calc_return(close_prices: pd.Series) -> pd.Series:
    returns = close_prices.apply(np.log) - close_prices.shift(1).apply(np.log)
    return returns


def add_derived_features(close: pd.Series) -> pd.DataFrame:
    """
    Input:
        close: pd.Series of prices (indexed by Date)
    Returns:
        features: pd.DataFrame with same index as `close` containing:
          - log_ret
          - return lags (1,2,5,10,20)
          - rolling mean/std of returns
          - price-based SMA/EMA (5,20), price/SMA ratios, price momentum
          - EWMA vol (20), absolute-return features
    """
    series = close.copy().astype(float)
    features = pd.DataFrame(index=series.index)

    # returns
    features["return"] = series - series.shift(1)
    features["log_ret"] = calc_return(series)
    features["ret_abs"] = features["log_ret"].abs()

    # # return lags (stationary features)
    # for k in [1, 2, 5, 10, 20]:
    #     features[f"lag_{k}"] = features["log_ret"].shift(k)

    # # rolling stats on returns (backward-looking)
    features["ret_ma_5"] = features["log_ret"].rolling(5, min_periods=1).mean()
    features["ret_ma_20"] = features["log_ret"].rolling(20, min_periods=1).mean()
    features["ret_std_20"] = features["log_ret"].rolling(20, min_periods=1).std()

    # # EWMA volatility (gives more weight to recent returns)
    # features["ewm_vol_20"] = features["log_ret"].ewm(span=20, adjust=False).std()

    # # price-based technicals (level features)
    features["sma_5"] = series.rolling(5, min_periods=1).mean()
    features["sma_20"] = series.rolling(20, min_periods=1).mean()
    # features["ema_12"] = s.ewm(span=12, adjust=False).mean()
    # features["price_vs_sma5"] = s / features["sma_5"] - 1.0  # relative to sma
    # features["price_vs_sma20"] = s / features["sma_20"] - 1.0
    # features["mom_5"] = s - s.shift(5)  # price momentum over 5 days
    # features["mom_20"] = s - s.shift(20)

    # additional useful transforms
    features["ret_roll_abs_mean_5"] = (
        features["ret_abs"].rolling(5, min_periods=1).mean()
    )
    features["ret_roll_abs_mean_20"] = (
        features["ret_abs"].rolling(20, min_periods=1).mean()
    )

    features = features.dropna().reset_index(drop=True)

    return features
