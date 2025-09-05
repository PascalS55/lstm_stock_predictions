import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from estimators.lstm import tune_model
from polygon import RESTClient
from post.eval_performance import log_returns_to_price, rescale_predictions
from prepro.data_curation import (
    create_windows,
    fetch_polygon_data,
    prepare_data,
    split_training_data,
)
from utils.helpers import ensure_dirs, read_stock_data
from utils.plot import plot_loss, plot_predictions


def main():
    # Load data
    current_stock = "aapl".upper()
    ensure_dirs(current_stock)
    file = os.path.join(
        os.path.dirname(__file__), "..", "archive", "Stocks", f"{current_stock}.us.txt"
    )

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", current_stock, "lstm_model.keras"
    )
    # if os.path.exists(model_path):
    #     print(f"Model for {current_stock} already exists at {model_path}.")
    #     user_input = (
    #         input("Model already exists. Do you want to retrain? (y/n): ")
    #         .strip()
    #         .lower()
    #     )
    #     if user_input != "y":
    #         print("Exiting without retraining.")
    #         return
    print(10 * "=", "Initiating training for", current_stock, 10 * "=")

    data = read_stock_data(file)
    print(data.head())

    # Define window size and horizon
    num_timesteps = 14  # here: days
    horizon = 7  # days ahead to predict

    print(10 * "=", "Preprocessing", 10 * "=")
    X, y = prepare_data(data, horizon=horizon)
    print(X.shape)

    # Split data into train, dev, and test sets
    test_size = min(int(0.15 * X.shape[0]), 10000)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_training_data(
        X, y, test_size=test_size, validate=True
    )

    # Scale X, avoid data leakage
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_dev = scaler.transform(X_dev)
    X_scaled_test = scaler.transform(X_test)

    # Windowing
    X_win_train, y_train = create_windows(
        X_scaled_train, y_train, window_size=num_timesteps
    )
    X_win_dev, y_dev = create_windows(X_scaled_dev, y_dev, window_size=num_timesteps)
    X_win_test, y_test = create_windows(
        X_scaled_test, y_test, window_size=num_timesteps
    )
    print("Shape X:", X_win_train.shape)

    subset_size = min(int(0.2 * X.shape[0]), 25000)
    subset_X = X_win_train[:subset_size, :, :]
    subset_y = y_train[:subset_size]
    subset_val = (
        X_win_train[subset_size : int(1.5 * subset_size), :, :],
        y_train[subset_size : int(1.5 * subset_size)],
    )

    tuned_lstm, best_hps = tune_model(
        "lstm",
        subset_X,
        subset_y,
        subset_val,
        max_trials=10,
        max_epochs=15,
        horizon=y.shape[1],
    )

    history = tuned_lstm.fit(
        X_win_train, y_train, validation_data=(X_win_dev, y_dev), epochs=15
    )
    tuned_lstm.save(f"models/{current_stock}/lstm_model.keras")
    with open(f"models/{current_stock}/preprocessing_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Model and preprocessing scaler saved for {current_stock}")

    predictions = tuned_lstm.predict(X_win_test)
    plot_loss(history, current_stock)

    predicted_prices = log_returns_to_price(
        predictions[:, 0], data["Close"].iloc[-test_size - 1]
    )
    true_prices = log_returns_to_price(y_test[:, 0], data["Close"].iloc[-test_size - 1])

    results = pd.DataFrame(
        {
            key: value
            for i in range(predictions.shape[1])
            for key, value in [
                (f"pred_{i+1}", predictions[:, i]),
                (f"true_{i+1}", y_test[:, i]),
            ]
        }
    )
    results.to_csv(f"results/{current_stock}/lstm_predictions.csv", index=False)

    plot_predictions(true_prices, predicted_prices, current_stock)
    acc = r2_score(y_test[:, 0], predictions[:, 0])
    print(f"R^2 Score: {acc}")

    # Second training phase with new data
    print(10 * "=", "Second Training Phase", 10 * "=")
    load_dotenv(
        "C:\\Users\\pasca\\Documents\\Playground\\AlgoTrading\\local_secrets.env"
    )
    api_key = os.getenv("POLYGON_API_KEY")

    client = RESTClient(api_key)
    chart_data = fetch_polygon_data(client, current_stock)
    chart_data.to_csv(f"archive/polygon/{current_stock}_latest.csv", index=False)
    X_update, y_update = prepare_data(chart_data, horizon=horizon)

    # Transform data
    X_up_scaled = scaler.transform(X_update)

    X_up_pre, y_up_pre = create_windows(
        X_up_scaled, y_update, window_size=num_timesteps
    )

    X_train2, X_dev2, X_test2, y_train2, y_dev2, y_test2 = split_training_data(
        X_up_pre,
        y_up_pre,
        test_size=int(X_up_pre.shape[0] * 0.15),
        validate=True,
    )

    history = tuned_lstm.fit(
        X_train2, y_train2, validation_data=(X_dev2, y_dev2), epochs=15
    )
    tuned_lstm.save(f"models/{current_stock}/lstm_model.keras")
    print(f"Updated Model saved for {current_stock}")

    predictions = tuned_lstm.predict(X_test2)
    plot_loss(history, current_stock, update=True)

    y_true2 = log_returns_to_price(
        y_test2, chart_data["close"].iloc[-X_test2.shape[0] - 1]
    )
    y_pred2 = log_returns_to_price(
        predictions, chart_data["close"].iloc[-X_test2.shape[0] - 1]
    )

    results = pd.DataFrame(
        {
            key: value
            for i in range(predictions.shape[1])
            for key, value in [
                (f"pred_{i+1}", predictions[:, i]),
                (f"true_{i+1}", y_test2[:, i]),
            ]
        }
    )
    results.to_csv(f"results/{current_stock}/lstm_predictions_poly.csv", index=False)

    plot_predictions(y_true2[:, 0], y_pred2[:, 0], current_stock, update=True)
    acc = r2_score(y_test2[:, 0], predictions[:, 0])
    print(f"R^2 Score: {acc}")

    print(10 * "=", "Training complete for", current_stock, 10 * "=")


if __name__ == "__main__":
    main()
