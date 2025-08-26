from keras import Model, layers, callbacks, optimizers
import keras_tuner as kt
import time


def tune_model(
    model_type: str,
    x,
    y,
    val,
    max_trials=10,
    executions_per_trial=1,
    max_epochs=30,
    project_name="rnn_tuning",
) -> tuple[Model, kt.HyperParameters]:
    """
    Tunes LSTM or GRU model using Keras Tuner.

    Args:
        model_type (str): 'lstm' or 'gru'.
        x (np.ndarray): Training input data.
        y (np.ndarray): Training target data.
        val (tuple): (val_x, val_y).
        max_trials (int): How many different models to try.
        executions_per_trial (int): How many times to train each model for averaging.
        max_epochs (int): Number of training epochs for each trial.
        project_name (str): Tuner project folder name.

    Returns:
        best_model (Model): Best tuned model.
        best_hps (HyperParameters): Best hyperparameter set.
    """

    print(10 * "=", f"Tuning {model_type.upper()}", 10 * "=")
    num_timesteps = x.shape[1]
    num_features = x.shape[2]

    def model_builder(hp: kt.HyperParameters) -> Model:
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        dropout = hp.Float("dropout", 0.0, 0.5, step=0.1)
        learning_rate = hp.Float("lr", 1e-4, 1e-2, sampling="log")

        inputs = layers.Input(shape=(num_timesteps, num_features))

        if model_type.lower() == "lstm":
            x_rnn = layers.LSTM(units)(inputs)
        elif model_type.lower() == "gru":
            x_rnn = layers.GRU(units)(inputs)
        else:
            raise ValueError("model_type must be either 'lstm' or 'gru'")

        x = layers.Dropout(dropout)(x_rnn)
        outputs = layers.Dense(1)(x)

        model = Model(
            inputs=inputs, outputs=outputs, name=f"Tuned_{model_type.upper()}"
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=["mean_absolute_error", "mean_squared_error"],
        )
        return model

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="keras_tuning",
        project_name=project_name,
        overwrite=True,
        seed=42,
    )

    start = time.time()
    tuner.search(x, y, epochs=max_epochs, validation_data=val, verbose=1)
    end = time.time()
    print(f"Tuning {model_type} completed in: {(end - start) / 60:.2f} minutes")

    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_model: Model = tuner.hypermodel.build(best_hps)
    best_model.compile(
        optimizer=optimizers.Adam(learning_rate=best_hps.get("lr")),
        loss="mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"],
    )
    print("Best hyperparameters:", best_hps.values)
    print("Best model summary:")
    best_model.summary()

    return best_model, best_hps
