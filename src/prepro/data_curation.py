from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="linear"):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assume X is a DataFrame or convert to one
        X_df = pd.DataFrame(X).interpolate(
            method=self.method, limit_direction="forward"
        )
        return X_df.values


class TimeSeriesWindowizer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=10, target_shift=1):
        """
        window_size: number of past timesteps to include in each sample
        target_shift: how far ahead the target is relative to the end of the window
                      e.g., 1 = next step, 0 = last step in window
        """
        self.window_size = window_size
        self.target_shift = target_shift

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.array(X)
        X_windows, y_windows = self._create_windows(X, y)
        return X_windows if y_windows is None else (X_windows, y_windows)

    def _create_windows(self, X, y):
        Xs, ys = [], []
        end_index = len(X) - self.window_size - self.target_shift + 1
        for i in range(end_index):
            Xs.append(X[i : i + self.window_size])
            if y is not None:
                ys.append(y[i + self.window_size + self.target_shift - 1])
        Xs = np.array(Xs)
        ys = np.array(ys) if ys else None
        return Xs, ys


def create_preprocessing_pipeline(
    impute_method="linear", lof_neighbors=20, window_size=10, target_shift=1
):
    imputer = TimeSeriesImputer(method=impute_method)

    # outlier_detector = LocalOutlierFactor(n_neighbors=lof_neighbors)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    windowizer = TimeSeriesWindowizer(
        window_size=window_size, target_shift=target_shift
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            # ("outlier_detector", outlier_detector),
            ("scaler", scaler),
            ("windowizer", windowizer),
        ]
    )

    return pipeline


def prepare_data(data: pd.DataFrame, window_size: int = 14) -> tuple:
    correlation = data.corr()
    print(correlation["Close"].sort_values(ascending=False))

    X = (
        data[["Open", "High", "Low", "Volume"]]
        # .rolling(window=window_size)
        # .mean()
        .to_numpy()
    )
    y = data["Close"].to_numpy()
    # Remove NaN values from X and y
    # X = X[~np.isnan(y)]
    # y = y[~np.isnan(y)]
    y = y.reshape(-1, 1)
    return X, y
