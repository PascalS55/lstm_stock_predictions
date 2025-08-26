import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from estimators.lstm import tune_model
from polygon import RESTClient
from post.eval_performance import rescale_predictions
from prepro.data_curation import (
    create_preprocessing_pipeline,
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
    X, y = prepare_data(data)
    print(X.shape)

    num_timesteps = 14  # here: days
    pipeline = create_preprocessing_pipeline(window_size=num_timesteps)
    print(10 * "=", "Preprocessing", 10 * "=")
    processed_data = pipeline.fit_transform(X)
    print("Shape X:", processed_data.shape)

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y = y_scaler.fit_transform(y)
    print("Shape y:", y.shape)

    subset_size = min(int(0.2 * processed_data.shape[0]), 25000)
    subset_X = processed_data[:subset_size, :, :]
    subset_y = y[num_timesteps : subset_size + num_timesteps]
    subset_val = (
        processed_data[subset_size : int(1.5 * subset_size), :, :],
        y[(subset_size + num_timesteps) : num_timesteps + int(1.5 * subset_size)],
    )

    tuned_lstm, best_hps = tune_model(
        "lstm", subset_X, subset_y, subset_val, max_trials=10, max_epochs=15
    )

    # Split data into train, dev, and test sets
    test_size = min(int(0.15 * processed_data.shape[0]), 10000)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_training_data(
        processed_data, y[num_timesteps:], test_size=test_size, validate=True
    )

    history = tuned_lstm.fit(
        X_train, y_train, validation_data=(X_dev, y_dev), epochs=15
    )
    tuned_lstm.save(f"models/{current_stock}/lstm_model.keras")
    with open(f"models/{current_stock}/preprocessing_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model and preprocessing pipeline saved for {current_stock}")

    predictions = tuned_lstm.predict(X_test)
    plot_loss(history, current_stock)

    y_true, y_pred = rescale_predictions(y_test, predictions, y_scaler)
    plot_predictions(y_true, y_pred, current_stock)

    acc = r2_score(y_true, y_pred)
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
    X_update, y_update = prepare_data(chart_data)

    # Transform data
    X_up_pre = pipeline.transform(X_update)
    target_updated = y_scaler.transform(y_update)

    X_train2, X_dev2, X_test2, y_train2, y_dev2, y_test2 = split_training_data(
        X_up_pre,
        target_updated[num_timesteps:],
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

    y_true2, y_pred2 = rescale_predictions(y_test2, predictions, y_scaler)
    plot_predictions(y_true2, y_pred2, current_stock, update=True)

    acc = r2_score(y_true2, y_pred2)
    print(f"R^2 Score: {acc}")

    print(10 * "=", "Training complete for", current_stock, 10 * "=")


if __name__ == "__main__":
    main()
