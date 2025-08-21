from sklearn.metrics import r2_score
from utils.helpers import ensure_dirs, read_stock_data
from utils.plot import plot_predictions, plot_loss
from prepro.data_curation import create_preprocessing_pipeline, prepare_data
from estimators.lstm import tune_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pickle


def main():
    # Load data
    current_stock = "aapl"
    ensure_dirs(current_stock)
    file = os.path.join(
        os.path.dirname(__file__), "..", "archive", "Stocks", f"{current_stock}.us.txt"
    )

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
    y = y_scaler.fit_transform(y.reshape(-1, 1))
    target = pipeline.named_steps["windowizer"].transform(y)
    print("Shape y:", target.shape)

    subset_size = min(int(0.2 * processed_data.shape[0]), 25000)
    subset_X = processed_data[:subset_size, :, :]
    subset_y = y[num_timesteps : subset_size + num_timesteps]
    subset_val = (
        processed_data[subset_size : int(1.5 * subset_size), :, :],
        y[(subset_size + num_timesteps) : num_timesteps + int(1.5 * subset_size)],
    )

    tuned_lstm, best_hps = tune_model(
        "lstm", subset_X, subset_y, subset_val, 10, max_epochs=15
    )

    # Split data into train, dev, and test sets
    test_size = min(int(0.15 * processed_data.shape[0]), 10000)
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, y[num_timesteps:], test_size=test_size, shuffle=False
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=test_size, shuffle=False
    )
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Dev shape: {X_dev.shape}, {y_dev.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")

    tuned_lstm.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=15)
    tuned_lstm.save(f"models/{current_stock}/lstm_model.keras")
    with open(f"models/{current_stock}/preprocessing_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model and preprocessing pipeline saved for {current_stock}")

    predictions = tuned_lstm.predict(X_test)
    plot_loss(tuned_lstm.history, current_stock)

    y_pred = y_scaler.inverse_transform(predictions)
    y_true = y_scaler.inverse_transform(y_test)
    plot_predictions(y_true, y_pred, current_stock)

    acc = r2_score(y_true, y_pred)
    print(f"R^2 Score: {acc}")


if __name__ == "__main__":
    main()
