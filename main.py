# Standard packages
import pandas as pd

# Scripts
from preprocess import config, get_timestamps, collect_data, plot_closing, plot_gain, compare_stocks
from models import TorchRNN, rnn_params, transf_params, TransformerModel
from dataset import GetDataset
from train import Classifier, plot_predictions
import Analysis


def visualization():
    for idx, stock in enumerate(config.stock_names):
        timestamps = get_timestamps(config.yrs, config.mths, config.dys)
        df = collect_data(timestamps, stock, config.moving_averages, True)
        fig1 = plot_closing(df, moving_averages=True, intervals=None)
        fig1.show()
        fig2 = plot_gain(df)
        fig2.show()
        daily_returns, fig1_c, fig2_c = compare_stocks(config.stock_names_compare, timestamps)


def run(stock: str, model_type: str, stationary=True):
    df = Analysis.get_data(stock)
    df["Company stock name"] = stock.split('/')[-1].split('.')[0]
    dataset = GetDataset(df)
    dataset.get_dataset(scale=False, stationary=stationary)
    train_data, test_data, train_data_len = dataset.split(train_split_ratio=0.8, time_period=30)
    train_data, test_data = dataset.get_torchdata()
    x_train, y_train = train_data
    x_test, y_test = test_data

    if model_type == 'lstm':
        params = rnn_params
        model = TorchRNN(rnn_type=params.rnn_type, input_dim=params.input_dim,
                         hidden_dim=params.hidden_dim, output_dim=params.output_dim,
                         num_layers=params.num_layers)
    elif model_type == 'transformer':
        params = transf_params
        model = TransformerModel(params)
    else:
        raise ValueError('Wrong model type selection, select either "rnn" or "transformer"!')

    clf = Classifier(model)
    clf.train([x_train, y_train], params=params)
    y_scaler = dataset.y_scaler
    predictions = clf.predict([x_test, y_test], y_scaler, data_scaled=False)
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    predictions.index = df.index[-len(x_test):]
    predictions['Actual'] = y_test[:-1]
    predictions.rename(columns={0: 'Predictions'}, inplace=True)
    if stationary:
        predictions = Analysis.inverse_stationary_data(old_df=df, new_df=predictions,
                                                       orig_feature='Actual', new_feature='Predictions',
                                                       diff=12, do_orig=False)
    plot_predictions(df, train_data_len, predictions["Predictions"].values, model_type)


if __name__ == '__main__':
    # visualization()
    run('./Data/AMZN.csv', 'transformer', True)
