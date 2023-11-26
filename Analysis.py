# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style('darkgrid')
plt.style.use('seaborn-darkgrid')


# statistics
import statsmodels.api as sm

# Collect data from yahoo finance
from pandas_datareader.data import DataReader
# For time stamps
from datetime import datetime

# scripts
import preprocess


class analysis_config:
    lookback_days = -1
    season_period = 365

def get_data(csv):
    if csv:
        df = pd.read_csv(csv)
        df = df.set_index('Date')
        return df
    else:
        timestamps = preprocess.get_timestamps(preprocess.config.yrs,
                                               preprocess.config.mths,
                                               preprocess.config.dys)

        df = preprocess.collect_data(timestamps, preprocess.config.stock_names[0],
                                     moving_averages=preprocess.config.moving_averages,
                                    include_gain=True)
        return df

def data_preparation(df):
    # Get inter-daily mean
    df["Mean"] = (df["Low"] + df["High"]) / 2
    # Creating target column of next day price based on 'lookback_days' previous days
    df_new = df.copy()
    df_new["Actual"] = df_new["Mean"].shift(analysis_config.lookback_days).dropna()

    # If date is not index yet
    if df_new.columns[0] == 'Date':
        df_new["Date"] = pd.to_datetime(df_new["Date"])
        df_new.index = df_new["Date"]

    return df_new

def analyse_movement(df):
    # Get the trend, seasonality and noise of the data
    seas_d = sm.tsa.seasonal_decompose(df["Mean"], model="add", period=analysis_config.season_period)
    seas_trend = seas_d.trend.values
    seas_seas = seas_d.seasonal.values
    seas_res = seas_d.resid.values
    seas_obs = seas_d.observed.values
    dates = [str(seas_d.trend.index.values[i]).split('T')[0] for i in range(len(seas_d.trend.index.values))]

    matplotlib.style.use('seaborn-darkgrid')
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    first_plot_labels = [item.get_text() for item in axes[0].get_xticklabels()]
    nr_label_indexes = len(dates) // (len(first_plot_labels))
    first_plot_labels_new = [str(dates[0])]
    [first_plot_labels_new.append(dates[i * nr_label_indexes]) for i in range(len(first_plot_labels))]
    first_plot_labels_new.append(dates[-1])
    axes[0].set_xticklabels(first_plot_labels_new)
    axes[0].plot(seas_obs, color='#76C2F7', linewidth=1)
    axes[0].set_title('Observed data')
    axes[0].set_ylabel('USD $')

    second_plot_labels = [item.get_text() for item in axes[1].get_xticklabels()]
    nr_label_indexes = len(dates) // (len(second_plot_labels)-2)
    second_plot_labels_new = [str(dates[0])]
    [second_plot_labels_new.append(dates[i * nr_label_indexes]) for i in range(len(second_plot_labels)-2)]
    second_plot_labels_new.append(dates[-1])
    axes[1].set_xticklabels(second_plot_labels_new)
    axes[1].plot(seas_trend, color='#9876F7', linewidth=1)
    axes[1].set_ylabel('Trend')

    third_plot_labels = [item.get_text() for item in axes[2].get_xticklabels()]
    nr_label_indexes = len(dates) // (len(third_plot_labels))
    third_plot_labels_new = [str(dates[0])]
    [third_plot_labels_new.append(dates[i * nr_label_indexes]) for i in range(len(third_plot_labels))]
    third_plot_labels_new.append(dates[-1])
    axes[2].set_xticklabels(third_plot_labels_new)
    axes[2].plot(seas_seas, color='#F776DF', linewidth=1)
    axes[2].set_ylabel('Seasonality')

    fourth_plot_labels = [item.get_text() for item in axes[3].get_xticklabels()]
    nr_label_indexes = len(dates) // (len(fourth_plot_labels)-2)
    fourth_plot_labels_new = [str(dates[0])]
    [fourth_plot_labels_new.append(dates[i * nr_label_indexes]) for i in range(len(fourth_plot_labels)-2)]
    fourth_plot_labels_new.append(dates[-1])
    axes[3].set_xticklabels(fourth_plot_labels_new)
    axes[3].plot(seas_res, color='#F77676', linewidth=1)
    axes[3].set_ylabel('Random noise')
    plt.savefig('./demonstration_images/analysis_ex.png')
    plt.show()


def ADFtest(time_series):
    # Augmented Dickey-Fuller test to check if data is stationary
    dfout = {}
    dftest = sm.tsa.adfuller(time_series.dropna(), autolag='AIC', regression='ct')
    for key, val in dftest[4].items():
        dfout[f'critical value ({key})'] = val
    if dftest[1] <= 0.05:
        print(f"Strong evidence against Null Hypothesis, p-value: {dftest[1]:.4f} < 0.05")
        print("Reject Null Hypothesis - Data is Stationary")
        return True
    else:
        print("Strong evidence for Null Hypothesis")
        print(f"Accept Null Hypothesis - Data is not Stationary, p-value: {dftest[1]:.4f} > 0.05")
        return False

def get_stationary_data(df:pd.DataFrame, columns:list, diff:int):
    # Making the data stationary
    df_cp = df.copy()
    for col in columns:
        df_cp[str(col)] = pd.DataFrame(np.log(df_cp[str(col)]).diff().diff(diff))
    return df_cp

def inverse_stationary_data(old_df:pd.DataFrame, new_df: pd.DataFrame, orig_feature: str,
                            new_feature: str, diff: int, do_orig=True):
    # Inverse the stationary data transformation

    if do_orig:
        new_df[orig_feature] += np.log(old_df[orig_feature]).shift(1)
        new_df[orig_feature] += np.log(old_df[orig_feature]).diff().shift(diff)
        new_df[orig_feature] = np.exp(new_df[orig_feature])
    new_df[new_feature] += np.log(old_df[orig_feature]).shift(1)
    new_df[new_feature] += np.log(old_df[orig_feature]).diff().shift(diff)
    new_df[new_feature] = np.exp(new_df[new_feature])
    return new_df

def plot_stationary(df, df_stat):
    matplotlib.style.use('seaborn-darkgrid')
    fig, axes = plt.subplots(1,2,figsize=(20,6))
    axes[0].plot(df['Close'], color='#DC76F7', linewidth=1)
    axes[1].plot(df_stat['Close'], color='#DC76F7', linewidth=1)

    dates = [str(df.index.values[i]).split('T')[0] for i in range(len(df.index.values))]
    plot_labels = [item.get_text() for item in axes[0].get_xticklabels()]
    nr_label_indexes = len(dates) // (len(plot_labels)-1)
    plot_labels_new = [str(dates[0])]
    [plot_labels_new.append(dates[i * nr_label_indexes]) for i in range(len(plot_labels)-1)]
    plot_labels_new.append(dates[-1])
    axes[0].set_xticklabels(plot_labels_new)
    axes[1].set_xticklabels(plot_labels_new)
    axes[0].set_title('Original closing price')
    axes[1].set_title('Stationary data of original closing price')
    plt.savefig('./demonstration_images/stationary_data_demo.png')
    plt.show()


if __name__=='__main__':
    df = get_data('./Data/AAPL.csv')
    df_new = data_preparation(df)
    analyse_movement(df_new)
    is_stat = ADFtest(df_new["Close"])
    if not is_stat:
        df_stat = df_new.copy()
        df_stat = get_stationary_data(df_stat, ['Close'], 12)
        is_stat_new = ADFtest(df_stat["Close"])
        plot_stationary(df_new, df_stat)
