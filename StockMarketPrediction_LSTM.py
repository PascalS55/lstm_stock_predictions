import keras_tuner
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.api.layers import LSTM, Dense, Input
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from DataCuration import read_currently_used_data, read_stock_data


def create_lstm_model(input_shape):
    """
    Create a Sequential LSTM model.

    Parameters:
    - input_shape: tuple, the shape of the input data (time_steps, num_features)

    Returns:
    - model: a compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Input(input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))  # Assuming a single output
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def tune_model(build_model, xtrain, ytrain):
    # Create the tuner
    tuner = keras_tuner.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=5,
        executions_per_trial=3,
        directory="tuning/",
    )

    # Perform the search
    tuner.search(xtrain, ytrain, epochs=10, validation_split=0.2)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


def plot_predictions(ytest, ypred, name):
    """
    Plot the predicted values against the actual values.

    Parameters:
    - ytest: numpy array, the true values
    - ypred: numpy array, the predicted values
    """
    # Convert predictions to the same shape as ytest if necessary
    if ypred.ndim > ytest.ndim:
        ypred = ypred.squeeze()  # Remove extra dimensions

    # Create a new figure
    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.plot(ytest, color="blue", label="Actual Values")

    # Plot predicted values
    plt.plot(ypred, color="red", linestyle="dashed", label="Predicted Values")

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted Values")
    plt.legend()

    # Show and save the plot
    plt.savefig("figs/" + name)
    plt.show()


current_stock = "aapl"
file = "archive/Stocks/" + current_stock + ".us.txt"
print(
    10 * "=",
    "Starting with Model Training for:",
    file.split("/")[-1].split(".")[0].upper(),
    10 * "=",
)
data = read_stock_data(file)

correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))

x = data[["Open", "High", "Low", "Volume"]].to_numpy()
y = data["Close"].to_numpy()
y = y.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

input_shape = (xtrain.shape[1], 1)
model = create_lstm_model(input_shape)

model.fit(xtrain, ytrain, batch_size=1, epochs=12)
model.save("models/" + current_stock + ".keras")

ypred = model.predict(xtest)
acc = r2_score(ytest, ypred)
print(acc)

plot_predictions(ytest, ypred, current_stock)
