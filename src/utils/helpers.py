import os
import pandas as pd


def read_stock_data(file_path):
    """
    Reads stock or ETF data from a text file and returns a pandas DataFrame.

    :param file_path: str, path to the text file containing the stock data
    :return: DataFrame, contains the stock data
    """
    # Define column names based on the provided format
    column_names = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]

    # Read the data into a pandas DataFrame
    data = pd.read_csv(file_path, names=column_names, header=0, parse_dates=["Date"])
    # append additional parameters like moving averages, momentum strategies

    return data


def read_currently_used_data():
    """
    Reads all the ETFs and stocks listed in the currently_used.txt file.

    :return: dict, a dictionary with stock/ETF names as keys and their data as pandas DataFrames
    """
    currently_used_file = "currently_used.txt"  # Replace with the actual file path
    etf_folder = "archive/ETFs"  # Replace with the actual folder path
    stock_folder = "archive/Stocks"
    data_dict = {}

    with open(currently_used_file, "r") as file:
        lines = file.readlines()
        section = None
        for line in lines:
            line = line.strip()
            if line == "ETFs":
                section = "ETFs"
            elif line == "Stocks":
                section = "Stocks"
            elif line.startswith("-"):
                ticker = line[2:].lower()  # Convert ticker to uppercase
                file_path = os.path.join(
                    etf_folder if section == "ETFs" else stock_folder,
                    f"{ticker}.us.txt",
                )
                if os.path.exists(file_path):
                    data_dict[ticker] = read_stock_data(file_path)
                else:
                    print(f"File not found: {file_path}")

    return data_dict


def ensure_dirs(current_stock: str):
    """
    Ensures that the necessary directories for saving models and figures exist.
    """
    os.makedirs(f"models/{current_stock}", exist_ok=True)
    os.makedirs(f"figs/{current_stock}", exist_ok=True)
    os.makedirs(f"results/{current_stock}", exist_ok=True)
