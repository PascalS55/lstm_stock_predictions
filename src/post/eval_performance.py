from sklearn.preprocessing import FunctionTransformer


def rescale_predictions(y_true, y_pred, scaler: FunctionTransformer):
    y_true_rescaled = scaler.inverse_transform(y_true)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    return y_true_rescaled, y_pred_rescaled
