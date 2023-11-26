# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-darkgrid')
sns.set_style('darkgrid')

# statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# scripts
import Analysis


class ARIMA(object):
    def __init__(self, train_split: float, feature_of_interest: str):
        self.train_split = train_split
        self.feature = feature_of_interest

    def train_test_split(self, df):
        train_size = int(self.train_split * df.shape[0])

        df_train = pd.DataFrame(df[:train_size]).dropna()
        df_test = pd.DataFrame(df[train_size:])
        return df_train, df_test, train_size

    def AutoRegression(self, p, df):
        df_temp = df

        # Generating the lagged p terms
        for i in range(1, p + 1):
            df_temp['Shifted_values_%d' % i] = df_temp[self.feature].shift(i)

        # Split data into train/test
        df_train, df_test, train_size = self.train_test_split(df_temp)
        self.train_size = train_size
        self.test_size = len(df_temp) - train_size

        X_train = df_train.iloc[:, 1:].values.reshape(-1, p)
        y_train = df_train.iloc[:, 0].values.reshape(-1, 1)
        X_test = df_test.iloc[:, 1:].values.reshape(-1, p)

        # Linear regression to get the coefficents of lagged terms
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_
        df_train['Predicted'] = X_train.dot(lr.coef_.T) + lr.intercept_
        df_test['Predicted'] = X_test.dot(lr.coef_.T) + lr.intercept_

        # Loss
        RMSE = np.sqrt(mean_squared_error(df_test[self.feature], df_test['Predicted']))

        #print("The RMSE is :", RMSE, ", Value of p : ", p)
        return [df_train, df_test, theta, intercept, RMSE]

    def MovingAverage(self, q, res):
        for i in range(1, q + 1):
            res['Shifted_values_%d' % i] = res['Residuals'].shift(i)

        res_train, res_test, train_size = self.train_test_split(res)

        X_train = res_train.iloc[:, 1:].values.reshape(-1, q)
        y_train = res_train.iloc[:, 0].values.reshape(-1, 1)
        X_test = res_test.iloc[:, 1:].values.reshape(-1, q)

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_

        res_train['Predicted'] = X_train.dot(lr.coef_.T) + lr.intercept_
        res_test['Predicted'] = X_test.dot(lr.coef_.T) + lr.intercept_

        RMSE = np.sqrt(mean_squared_error(res_test['Residuals'], res_test['Predicted']))

        #print("The RMSE is :", RMSE, ", Value of q : ", q)
        return [res_train, res_test, theta, intercept, RMSE]

    def run(self, df_orig: pd.DataFrame, make_stationary: bool):
        df = df_orig

        # Make data stationary if not done
        if make_stationary:
            df = Analysis.get_stationary_data(df, [self.feature], 12).dropna()

        best_RMSE = 1e5
        best_p = -1

        for i in range(1, 25):
            [df_train, df_test, theta, intercept, RMSE] = self.AutoRegression(i, pd.DataFrame(df.Close))
            if (RMSE < best_RMSE):
                best_RMSE = RMSE
                best_p = i

        # print(best_p)
        [df_train, df_test, theta, intercept, RMSE] = self.AutoRegression(best_p, pd.DataFrame(df.Close))

        df_c = pd.concat([df_train, df_test])

        res = pd.DataFrame()
        res['Residuals'] = df_c.Close - df_c.Predicted

        best_RMSE = 1e5
        best_q = -1

        for i in range(1, 25):
            [res_train, res_test, theta, intercept, RMSE] = self.MovingAverage(i, pd.DataFrame(res.Residuals))
            if (RMSE < best_RMSE):
                best_RMSE = RMSE
                best_q = i

        # print(best_q)
        [res_train, res_test, theta, intercept, RMSE] = self.MovingAverage(best_q, pd.DataFrame(res.Residuals))
        # print(theta)
        # print(intercept)

        res_c = pd.concat([res_train, res_test])
        df_c.Predicted += res_c.Predicted
        f = False
        if f:
            df_c.Close += np.log(df_orig).shift(1).Close
            df_c.Close += np.log(df_orig).diff().shift(12).Close
            df_c.Predicted += np.log(df_orig).shift(1).Close
            df_c.Predicted += np.log(df_orig).diff().shift(12).Close
            df_c.Close = np.exp(df_c.Close)
            df_c.Predicted = np.exp(df_c.Predicted)

        if make_stationary:
            df_c = Analysis.inverse_stationary_data(df_orig, df_c, 'Close', 'Predicted', 12)
        return df_c, res

    def plot_results(self, df, res, stock_name: str, save_plot=True):
        matplotlib.style.use('seaborn-darkgrid')
        RMSE = np.sqrt(mean_squared_error(df.iloc[self.train_size:, :]["Predicted"].values, df.iloc[self.train_size:, :]["Close"].values))
        df.iloc[self.train_size:, :][["Close", "Predicted"]].plot(figsize=(20, 8), xlabel='Date', ylabel='Price USD ($)',
                                        title=f'Final predictions from ARIMA model for "{stock_name}" stock',
                                        color=["#c6e2ff", "#deaddd"], linewidth=1.5)
        if save_plot:
            plt.savefig('./demonstration_images/arima_predictions.png')
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.kdeplot(res["Residuals"], color="#c7c6ff", shade=True, linewidth=2)
        plt.title(f'Density of the residuals from the auto-regression for "{stock_name}" stock')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        if save_plot:
            #fig_r = ax_r.get_figure()
            plt.savefig('./demonstration_images/arima_residuals.png')
        plt.show()


if __name__=='__main__':
    df = Analysis.get_data('./Data/AMZN.csv')
    df = df[["Close"]]
    df.rename(columns={0: 'Close'}, inplace=True)

    arima_model = ARIMA(0.8, 'Close')
    df_c, res = arima_model.run(df, True)

    arima_model.plot_results(df_c, res, 'AMZN', save_plot=True)
