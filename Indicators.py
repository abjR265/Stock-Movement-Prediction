import numpy as np
import pandas as pd

def daily_return(df: pd.DataFrame):
    df['Daily_returns'] = df['Close'].pct_change(1).fillna(0)
    return df

### MOMENTUM INDICATORS ###
def roc_indicator(df: pd.DataFrame):
    # Computes rate of change (RoC), i.e momentum - percent change
    df["RoC"] = df['Close'].diff() / df['Close'][:-1]
    return df

def williams_r(df: pd.DataFrame, lookback: int):
    # Computes Williams %R that measures overbought and oversold levels
    wr = np.zeros(len(df))
    for t_idx in range(len(df)):
        if t_idx + 1 <= lookback:
            wr[t_idx] = 0
        else:
            highest = np.max(df['High'][t_idx-lookback:t_idx].values)
            lowest = np.min(df['Low'][t_idx-lookback:t_idx].values)
            wr[t_idx] = (highest - df['Close'][t_idx]) / (highest - lowest)
    df["williams_r"] = wr
    return df

### VOLUME INDICATORS ###
def money_flow_index(df: pd.DataFrame, period: int):
    # Measures buying and selling pressure (if below 20 then buy if above 80 then sell)
    typical_price = (df['Close'] + df['High'] + df['Low']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow, negative_flow = [], []
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i-1])
            negative_flow.append(0)
        elif typical_price[i] < typical_price[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow[i-1])
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    positive_mf = [sum(positive_flow[i + 1 - period:i + 1]) for i in range(period-1, len(positive_flow))]
    negative_mf = [sum(negative_flow[i + 1 - period:i + 1]) for i in range(period - 1, len(negative_flow))]
    idx = 0
    mfi = np.zeros(len(df))
    for t_idx in range(len(df)):
        if t_idx + 1 <= period:
            mfi[t_idx] = 0
        else:
            mfi[t_idx] = 100 * (positive_mf[idx] / (positive_mf[idx] + negative_mf[idx]))
            idx += 1
    df["MFI"] = mfi
    return df

### VOLATILITY INDICATORS ###
def ulcer_index(df: pd.DataFrame, lookback: int):
    # Measures downside risk in terms of depth and duration of price declines
    ui = np.zeros(len(df))
    for t_idx in range(len(df)):
        if t_idx + 1 <= lookback:
            ui[t_idx] = 0
        else:
            maxprice = np.max(df['Close'][t_idx-lookback:t_idx].values)
            percentage_drawdown = [(df['Close'][t_idx-i]-maxprice)/maxprice * 100 for i in reversed(range(lookback))]
            ulcer_ind = np.sqrt(np.sum(np.array(percentage_drawdown)**2) / lookback)
            ui[t_idx] = ulcer_ind
    df['Ulcer_index'] = ui
    return df

def average_true_range(df: pd.DataFrame, lookback: int):
    # Measures market volatility
    av_tr_rang = np.zeros(len(df))
    for t_idx in range(len(df)):
        if t_idx + 1 <= lookback:
            av_tr_rang[t_idx] = 0
        else:
            true_ranges = []
            for idx in reversed(range(lookback)):
                tr1 = df['High'][t_idx-idx] - df['Low'][t_idx-idx]
                tr2 = np.abs(df['High'][t_idx-idx] - df['Close'][t_idx-idx])
                tr3 = np.abs(df['Low'][t_idx-idx] - df['Close'][t_idx-idx])
                true_ranges.append(np.max([tr1, tr2, tr3]))
            atr = sum(true_ranges) / lookback
            av_tr_rang[t_idx] = atr
    df['ATR'] = av_tr_rang
    return df

### TREND INDICATORS ###

def simple_moving_average(df: pd.DataFrame, windows: list):
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean().fillna(0)
    return df

def exponential_moving_average(df: pd.DataFrame, windows: list):
    for window in windows:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean().fillna(0)
    return df


def get_indicators(df: pd.DataFrame):
    df = daily_return(df)
    df = roc_indicator(df)
    df = williams_r(df, 14)
    df = money_flow_index(df, 14)
    df = ulcer_index(df, 14)
    df = average_true_range(df, 14)
    df = simple_moving_average(df, [5, 10, 20])
    df = exponential_moving_average(df, [20, 50])
    return df
