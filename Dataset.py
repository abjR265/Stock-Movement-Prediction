
# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch
sns.set_style('darkgrid')
plt.style.use("fivethirtyeight")

import preprocess
import Analysis

class GetDataset(object):
    def __init__(self, df):
        super(GetDataset, self).__init__()
        self.df = df
        self.df["Next_day_closing_price"] = df["Close"].shift(-1).dropna()
        if self.df.columns[0] == 'Date':
            self.df = self.df.set_index('Date')
        self.df["Actual"] = self.df["Next_day_closing_price"]



    def get_dataset(self, scale=True, stationary=False, indicators=False):
        '''
            Input: scale - if to scale the input data
        '''
        x_df = self.df[["Close", "Open", "High", "Low", "Volume"]].dropna()[:-1]
        y_df = self.df["Next_day_closing_price"].dropna().fillna(0)


        x_processed_df = preprocess.preprocess_data(x_df).fillna(0)
        if stationary:
            for col in x_processed_df.columns:
                #if not Analysis.ADFtest(x_processed_df[col]):
                print("\nMaking data stationary...\n")
                x_processed_df = Analysis.get_stationary_data(x_processed_df, [col], 12)
                #Analysis.ADFtest(x_processed_df[col])

            y_df = Analysis.get_stationary_data(self.df, ["Next_day_closing_price"], 12)['Next_day_closing_price']
            y_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        #print(x_processed_df)
        x_processed_df.replace([np.inf, -np.inf], 0, inplace=True)

        self.x_data_values = x_processed_df.fillna(0).values[:-1]
        self.y_data_values = y_df.values[:-1].reshape(-1, 1)

        self.x_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.y_scaler = MinMaxScaler(feature_range=(-1, 1))

        if scale:
            self.x_data = self.x_scaler.fit_transform(self.x_data_values)
            self.y_data = self.y_scaler.fit_transform(self.y_data_values)
            #self.y_data = self.y_data_values
        else:
            self.x_data = self.x_data_values
            self.y_data = self.y_data_values


    def get_size(self):
        '''
            Output: returns the length of the dataset
        '''
        return len(self.x_data)


    def split(self, train_split_ratio=0.8, time_period=30):
        '''
            Input: train_split_ratio - percentage of dataset to be used for
                                       the training data (float)
                   time_period - time span in days to be predicted (in)

            Output: lists of the training and validation data (input values and target values)
                    size of the training data
        '''

        train_data_size = int(np.ceil(self.get_size() * train_split_ratio))
        x_train_data = self.x_data[:train_data_size]
        y_train_data = self.y_data[:train_data_size]

        x_train = [x_train_data[i-time_period:i] for i in range(time_period, len(x_train_data))]
        y_train = y_train_data[time_period:]


        self.y_train = np.array(y_train)
        self.x_train = np.array(x_train)
        #self.x_train = np.reshape(np.array(x_train), (np.array(x_train).shape[0], np.array(x_train).shape[1], 1))
        print(f'Shape of train data: (x, y) = ({np.shape(self.x_train)}, {np.shape(self.y_train)})')

        x_test_data = self.x_data[train_data_size - time_period:]
        y_test = self.y_data[train_data_size:]
        x_test = [x_test_data[i-time_period:i] for i in range(time_period, len(x_test_data))]

        self.y_test = np.array(y_test)
        self.x_test = np.array(x_test)
        #self.x_test = np.reshape(np.array(x_test), (np.array(x_test).shape[0], np.array(x_test).shape[1], 1))
        print(f'Shape of test data: (x, y) = ({np.shape(self.x_test)}, {np.shape(self.y_test)})')
        return [self.x_train, self.y_train], [self.x_test, self.y_test], train_data_size


    def get_torchdata(self):
        self.x_train_tensor = torch.from_numpy(self.x_train).type(torch.Tensor)
        self.x_test_tensor = torch.from_numpy(self.x_test).type(torch.Tensor)

        self.y_train_tensor = torch.from_numpy(self.y_train).type(torch.Tensor)
        self.y_test_tensor = torch.from_numpy(self.y_test).type(torch.Tensor)

        return [self.x_train_tensor, self.y_train_tensor], [self.x_test_tensor, self.y_test_tensor]





class GetDatasetOld(object):
    def __init__(self, df, feature='Adj Close'):
        super(GetDatasetOld, self).__init__()
        self.df = df
        self.feature = feature


    def get_dataset(self, scale=True):
        '''
            Input: scale - if to scale the input data
        '''
        data = self.df.filter([str(self.feature)])
        self.data_values = data.values
        if scale:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.dataset = self.scaler.fit_transform(self.data_values)

        else:
            self.dataset = self.data_values


    def get_size(self):
        '''
            Output: returns the length of the dataset
        '''
        return len(self.dataset)


    def split(self, train_split_ratio = 0.8, time_period = 30):
        '''
            Input: train_split_ratio - percentage of dataset to be used for
                                       the training data (float)
                   time_period - time span in days to be predicted (in)

            Output: lists of the training and validation data (input values and target values)
                    size of the training data
        '''
        train_data_size = int(np.ceil(self.get_size() * train_split_ratio))
        self.train_data = self.dataset[0:int(train_data_size), :]
        x_train, y_train = [], []
        for i in range(time_period, len(self.train_data)):
            x_train.append(self.train_data[i-time_period:i, 0])
            y_train.append(self.train_data[i, :])

        self.y_train = np.array(y_train)
        self.x_train = np.reshape(np.array(x_train), (np.array(x_train).shape[0], np.array(x_train).shape[1], 1))
        print(f'Shape of train data: (x, y) = ({np.shape(self.x_train)}, {np.shape(self.y_train)})')

        self.test_data = self.dataset[train_data_size - time_period:, :]
        x_test = []
        self.y_test = self.dataset[train_data_size:, :]
        for i in range(time_period, len(self.test_data)):
            x_test.append(self.test_data[i - time_period:i, 0])

        self.x_test = np.reshape(np.array(x_test), (np.array(x_test).shape[0], np.array(x_test).shape[1], 1))
        print(f'Shape of test data: (x, y) = ({np.shape(self.x_test)}, {np.shape(self.y_test)})')
        return [self.x_train, self.y_train], [self.x_test, self.y_test], train_data_size


    def get_torchdata(self):
        self.x_train_tensor = torch.from_numpy(self.x_train).type(torch.Tensor)
        self.x_test_tensor = torch.from_numpy(self.x_test).type(torch.Tensor)

        self.y_train_tensor = torch.from_numpy(self.y_train).type(torch.Tensor)
        self.y_test_tensor = torch.from_numpy(self.y_test).type(torch.Tensor)

        return [self.x_train_tensor, self.y_train_tensor], [self.x_test_tensor, self.y_test_tensor]
