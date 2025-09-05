from sklearn.preprocessing import FunctionTransformer
import numpy as np


def rescale_predictions(y_true, y_pred, scaler: FunctionTransformer):
    y_true_rescaled = scaler.inverse_transform(y_true)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    return y_true_rescaled, y_pred_rescaled


def log_returns_to_price(log_returns: np.ndarray, initial_price: float) -> np.ndarray:
    """
    Convert predicted log-returns back to price series.

    Args:
        y_pred (np.ndarray): shape (n_samples, horizon) or (n_samples,)
        initial_price (float): price at the timestep immediately before predictions start

    Returns:
        np.ndarray: reconstructed prices of shape (n_samples, horizon)
    """
    log_returns = np.atleast_2d(log_returns)
    return initial_price * np.exp(np.cumsum(log_returns, axis=1))
