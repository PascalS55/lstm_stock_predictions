# Stock Market Prediction with LSTM

This project uses Long Short-Term Memory (LSTM) neural networks to predict stock prices. The model is trained on historical stock data and aims to forecast future stock prices based on past trends.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
<!-- - [Results](#results)
- [Contributing](#contributing)
- [License](#license) -->

## Introduction

Stock market prediction is a challenging task due to the complexity and non-linearity of financial time series. This project leverages LSTM, a type of recurrent neural network (RNN), to model and predict stock prices based on historical data.

## Features

- Data preprocessing and normalization
- LSTM model for time series prediction
- Hyperparameter tuning with Keras Tuner
- Model evaluation and visualization of predictions
- Modular codebase for easy reusability with different datasets

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/PascalS55/lstm_stock_predictions.git
   cd lstm_stock_predictions
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv env
   # On Windows
   .\env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your dataset:**
   Stock market data from the past years until 2017 is found in the archive folder. It's in the following format:

   ```plaintext
   Date        Open     High      Low     Close    Volume  OpenInt
   1984-09-07  0.42388  0.42902  0.41874  0.42388  23220030  0
   ```

2. **Train the model:**
   The model is trained using the data split from scikit-learn library and keras for the actual model.

## Model Architecture

The LSTM model consists of the following layers:

- Two LSTM layers with 128 and 64 units, respectively.
- A Dense layer with 25 units.
- A final Dense layer with 1 unit for the output.
