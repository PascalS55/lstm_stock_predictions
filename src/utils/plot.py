import matplotlib.pyplot as plt


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
    plt.savefig("figs/" + name + "/predictions.png")
    plt.show()


def plot_loss(history, name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    # plt.ylim([0, 0.1])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("figs/" + name + "/loss.png")
    # plt.show()
