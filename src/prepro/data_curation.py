from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from polygon import RESTClient
import pandas as pd
import numpy as np

from prepro.feature_eng import add_derived_features


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


def create_windows(X: np.array, y: np.array, window_size: int) -> np.array:
    X = np.array(X)
    n_samples, n_features = X.shape
    n_windows = n_samples - window_size + 1
    if n_windows <= 0:
        raise ValueError("window_size is larger than the number of samples")

    windows = np.zeros((n_windows, window_size, n_features))
    y_aligned = np.zeros((n_windows, y.shape[1]))
    for i in range(n_windows):
        windows[i] = X[i : i + window_size]
        y_aligned[i] = y[i + window_size - 1]
    return windows, y_aligned


def prepare_data(data: pd.DataFrame, horizon=1) -> tuple:
    # Handle both 'Open' and 'open' column names (case-insensitive)
    cols = [
        col
        for col in data.columns
        if col.lower() in ["open", "high", "low", "volume", "close"]
    ]
    raw_features = data[cols]
    data["dow"] = data["Date"].dt.dayofweek

    y_col = [col for col in data.columns if col.lower() == "close"]

    features = add_derived_features(data[y_col[0]])

    targets = pd.DataFrame(
        {f"target_{h+1}": features["log_ret"].shift(-h) for h in range(horizon)}
    )

    model_df = pd.concat(
        [raw_features, features, targets],
        axis=1,
    )
    model_df = model_df.dropna(
        subset=features.columns.tolist() + targets.columns.tolist()
    )

    X = model_df[raw_features.columns.tolist() + features.columns.tolist()].to_numpy()
    y = model_df[targets.columns.tolist()].to_numpy()

    return X, y


def fetch_polygon_data(client: RESTClient, current_stock: str):
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    back_dated = (
        pd.Timestamp.now()
        .replace(year=pd.Timestamp.now().year - 2)
        .strftime("%Y-%m-%d")
    )
    print("Date Range for stage 2 training:", back_dated, today)

    try:
        history = client.get_aggs(
            ticker=current_stock.upper(),
            multiplier=1,
            timespan="day",
            from_=back_dated,
            to=today,
        )
        chart_data = pd.DataFrame(history)
        chart_data["Date"] = chart_data["timestamp"].apply(
            lambda x: pd.to_datetime(x, unit="ms")
        )
        print(chart_data.head())
        return chart_data

    except Exception as e:
        print(f"Error fetching history: {e}")
        return


def split_training_data(data, y, test_size=0.2, validate=False, shuffle=False):
    if validate:
        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=test_size, shuffle=shuffle
        )
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train, y_train, test_size=test_size, shuffle=shuffle
        )
        print(f"Train shape: {X_train.shape}, {y_train.shape}")
        print(f"Dev shape: {X_dev.shape}, {y_dev.shape}")
        print(f"Test shape: {X_test.shape}, {y_test.shape}")
        return X_train, X_dev, X_test, y_train, y_dev, y_test

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=test_size, shuffle=shuffle
        )
        print(f"Train shape: {X_train.shape}, {y_train.shape}")
        print(f"Test shape: {X_test.shape}, {y_test.shape}")
        return X_train, X_test, y_train, y_test
