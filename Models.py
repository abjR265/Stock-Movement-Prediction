
import torch.nn as nn
import torch
import os
import transformer

class ModelUtils:
    '''
    A utility class to save and load model weights
    '''
    def save_model(self, save_path, model):
        root, ext = os.path.splitext(save_path)
        if not ext:
            save_path = root + '.pth'
        try:
            torch.save(model.state_dict(), save_path)
            print(f'Successfully saved to model to "{save_path}"!')
        except Exception as e:
            print(f'Unable to save model, check save path!')
            print(f'Exception:\n{e}')
            return None

    def load_model(self, load_path, model):
        try:
            model.load_state_dict(torch.load(load_path))
            print(f'Successfully loaded the model from path "{load_path}"')

        except Exception as e:
            print(f'Unable to load the weights, check if different model or incorrect path!')
            print(f'Exception:\n{e}')
            return None

class rnn_params:
    rnn_type = 'lstm'
    input_dim = 16
    hidden_dim = 2048
    num_layers = 1
    output_dim = 1
    n_epochs = 100
    lr = 0.00001

class transf_params:
    n_layers = 11
    num_heads = 12
    model_dim = 16  # nr of features
    forward_dim = 2048
    output_dim = 1
    dropout = 0
    n_epochs = 100
    lr = 0.01

class TransformerModel(nn.Module):
    def __init__(self, params):
        super(TransformerModel, self).__init__()
        self.transf = transformer.TransformerModel(n_layers=params.n_layers,
                                                   num_heads=params.num_heads,
                                                   model_dim=params.model_dim,
                                                   forward_dim=params.forward_dim,
                                                   output_dim=16,
                                                   dropout=params.dropout)
        self.linear = nn.Linear(16, params.output_dim)
    def forward(self, x):
        transf_out = self.transf(x)
        out = self.linear(transf_out)
        return out


class TorchRNN(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_layers, output_dim):
        super(TorchRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise KeyError('Invalid RNN type, select "lstm" or "gru"!')
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        elif self.rnn_type == 'gru':
            out, (hn) = self.rnn(x, (h0.detach()))
        else:
            raise KeyError('Invalid RNN type, select "lstm" or "gru"!')
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out
