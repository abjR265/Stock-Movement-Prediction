# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

sns.set_style('darkgrid')
plt.style.use('seaborn-darkgrid')
import random

# statistics
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# preprocess
from sklearn.preprocessing import MinMaxScaler

# scripts
import Analysis


def get_train_test(df, train_split, scale=True, stationary=False):
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_out = MinMaxScaler(feature_range=(0, 1))
    scaled_input, scaler_output = None, None
    df_orig = df.copy()
    if scale:
        scaled_input = sc_in.fit_transform(df[["Low", "High", "Open", "Close", "Volume", "Adj Close", "Mean"]])
        scaled_input = pd.DataFrame(scaled_input)
        X = scaled_input
        X.rename(columns={0: "Low", 1: "High", 2: "Open", 3: "Close", 4: "Volume", 5: "Adj Close", 6: "Mean"}, inplace=True)
        X.index = df.index

        scaler_output = sc_out.fit_transform(df[["Actual"]])
        scaler_output = pd.DataFrame(scaler_output)
        y = scaler_output
        y.rename(columns={0: "Next day price"}, inplace=True)
        y.index = df.index
    elif stationary:
        stationary_data = Analysis.get_stationary_data(df, df.columns if df.columns[0] != 'Date' else df.columns[1:], 12)
        X = stationary_data[["Low", "High", "Open", "Close", "Volume", "Adj Close", "Mean"]]
        y = stationary_data[["Actual"]]
    else: pass

    train_size = int(len(df) * train_split)
    train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
    test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()

    return train_X, train_y, test_X, test_y, scaled_input, scaler_output, sc_out, df_orig


def plot_acf(y_test):
    matplotlib.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    fig = sm.tsa.graphics.plot_acf(y_test, lags=50, ax=ax[0])
    fig = sm.tsa.graphics.plot_pacf(y_test, lags=50, ax=ax[1])
    plt.show()


def get_best_arima(train_X, train_y):
    step_wise = auto_arima(train_y, exogenous=train_X, start_p=1, start_q=1,
                           max_p=7, max_q=7, d=1, max_d=7, trace=True,
                           error_action="ignore", suppress_warnings=True, stepwise=True)
    print(step_wise.summary())

def perform_arima(train_X, train_y, model_name, order):
    if model_name=='arima':
        model = ARIMA(train_y,
                        exog=train_X,
                        order=order,
                        enforce_invertibility=False, enforce_stationarity=False)

    elif model_name=='sarimax':
        model = SARIMAX(train_y,
                        exog=train_X,
                        order=order,
                        enforce_invertibility=False, enforce_stationarity=False)

    else:
        raise KeyError('Invalid model selection, choose either "arima" or "sarimax"!')

    results = model.fit()
    return model, results

def predict_arima(results, test_X, train_X, scaler_output, sc_out, df_orig=None, scale=True, stationary=True):
    predictions = results.predict(start=len(train_X), end=len(train_X)+len(test_X)-1, exog=test_X).values
    forecast = results.forecast(steps=len(test_X), exog=test_X)
    act_arr = np.array(df_orig["Actual"][-len(test_X):].values)
    if scale:
        act_arr = sc_out.inverse_transform(np.array(scaler_output.iloc[len(train_X):, 0].values).reshape(-1, 1))
        predictions = sc_out.inverse_transform(np.array(predictions).reshape(-1, 1))
    act = pd.DataFrame(act_arr)
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    predictions.index = test_X.index
    predictions['Actual'] = act.values
    predictions.rename(columns={0: 'Predictions'}, inplace=True)
   # print(predictions)
    if stationary:
        predictions = Analysis.inverse_stationary_data(old_df=df_orig, new_df=predictions,
                                                       orig_feature='Actual', new_feature='Predictions',
                                                       diff=12, do_orig=False)

    return predictions

def plot_predictions(predictions, stock_name: str):
    matplotlib.style.use('seaborn-darkgrid')
    RMSE = np.sqrt(mean_squared_error(predictions["Predictions"].values, predictions["Actual"].values))
    predictions["Actual"].plot(figsize=(20, 8), legend=True, color="#81A6FF", linewidth=1)
    predictions["Predictions"].plot(legend=True, color="#C281FF", linewidth=1, figsize=(20, 8))
    plt.ylabel('USD $')
    plt.title(f'SARIMAX predictions for "{stock_name}" stock')
    plt.savefig('./demonstration_images/sarimax_predictions.png')
    plt.show()


if __name__=='__main__':
    df = Analysis.get_data('./Data/AMZN.csv')

    df_new = Analysis.data_preparation(df)
    setup = get_train_test(df_new.dropna(), 0.8, scale=False, stationary=True)
    train_X, train_y, test_X, test_y, scaled_input, scaler_output, sc_out, df_orig = setup
    #plot_acf(test_y)

    #get_best_arima(train_X, train_y)

    model, results = perform_arima(train_X, train_y, 'sarimax', (1, 1, 1))
    predictions = predict_arima(results, test_X, train_X, scaler_output, sc_out, df_orig, scale=False)
    plot_predictions(predictions, "AMZN")
