# Standard packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import plotly.graph_objects as go
sns.set_style('darkgrid')
import os
os.environ['KMP_WARNINGS'] = 'off'


class Classifier(object):
    def __init__(self, model):
        self.model = model

    def train(self, train_data, params):
        '''
            Input: train_data - list of input values (numpy array) and target values
                                (numpy array) of training data
                   model - model to be trained
                   show_progress - if the training process is showed (boolean)

        '''
        self.x_train, self.y_train = train_data
        criterion = torch.nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        hist = np.zeros(params.n_epochs)
        self.model.train()
        for epoch in range(params.n_epochs):
            y_train_pred = self.model(self.x_train)
            loss = criterion(y_train_pred, self.y_train)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print(f'Epoch: {epoch+1}/{params.n_epochs}\tMSE loss: {loss.item():.5f}')
            hist[epoch] = loss.item()

        return hist


    def predict(self, test_data, scaler, data_scaled=True):
        '''
            Input: test_data - list of input values (numpy array) and target values
                               (numpy array) of validation data
                   scaler - scaler object to inversely scale predictions
                   data_scaled - if scaler were used in the preprocessing (boolean)

            Output: predictions - numpy array of the predicted values
        '''


        self.x_test, self.y_test = test_data
        self.model.eval()
        predictions = self.model(self.x_test).detach().numpy()
        if data_scaled:
            predictions = scaler.inverse_transform(predictions)

        return predictions

def plot_predictions(df, train_data_size, predictions, model_name:str):
    '''
        Input: df - dataframe of stock values
               train_data_size - length of the training data, number of elements (int)
               predictions - numpy array of the prdicted values
    '''
    colors = ['#579BF5', '#C694F6', '#F168F1']
    fig = go.Figure()
    train = df[:train_data_size]
    valid = df[train_data_size:][:-2]
    valid['Predictions'] = predictions
    RMSE = np.sqrt(mean_squared_error(predictions, valid['Adj Close'].values))
    x_train = [str(train.index[i]).split()[0] for i in range(len(train))]
    x_val = [str(valid.index[i]).split()[0] for i in range(len(valid))]

    fig.add_trace(
        go.Scatter(x=x_train, y=train['Adj Close'], mode='lines', line_color=colors[0], line_width=2,
                   name='Training data'))

    fig.add_trace(
        go.Scatter(x=x_val, y=valid['Adj Close'], mode='lines', line_color=colors[1], line_width=2,
                   name='Validation data'))

    fig.add_trace(
        go.Scatter(x=x_val, y=valid['Predictions'], mode='lines', line_color=colors[2], line_width=2,
                   name='Predictions'))

    fig.update_layout(showlegend=True)
    fig.update_layout(title=dict(text=f'Predictions of stock "{train["Company stock name"][0]}" from {x_val[0]} to {x_val[len(valid) - 1]}',
                                 xanchor='auto'),
                      xaxis=go.layout.XAxis(
                          title=go.layout.xaxis.Title(
                              text="Date")),
                      yaxis=go.layout.YAxis(
                          title=go.layout.yaxis.Title(
                              text="Adjusted closing price USD ($)"))
                      )
    fig.write_image(f'./demonstration_images/{model_name}_predictions.png')
    fig.show()
