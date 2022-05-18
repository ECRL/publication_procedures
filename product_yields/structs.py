from typing import Tuple, List, Dict

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class SoftmaxMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_hidden: int, output_dim: int, dropout: float = 0.0):
        """
        Feed-forward multilayer perceptron with softmax layer at output

        Args:
            input_dim (int): number of input features for model
            hidden_dim (int): number of neurons in hidden layers
            n_hidden (int): number of hidden layers
            output_dim (int): number of features per sample target
            dropout (float): random neuron dropout probability during training
        """

        super(SoftmaxMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(nn.Softmax(dim=1))
        self._dropout = dropout

    def forward(self, x: 'torch.tensor') -> 'torch.tensor':
        """
        Feed-forward operation

        Args:
            x (torch.tensor): input data, size [m, n_features]

        Returns:
            torch.tensor: fed-forward data, size [m, n_targets]
        """

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self._dropout, training=self.training)
        return self.layers[-1](x)


class ReactorDataset(Dataset):

    def __init__(self, X: 'numpy.array', y: 'numpy.array'):
        """
        Formats data into struct for batching/training

        Args:
            X (numpy.array): input variables, shape [n_samples, n_features]
            y (numpy.array): target variables, shape [n_samples, n_targets]
        """

        self.X = torch.as_tensor(X).type(torch.float32)
        self.y = torch.as_tensor(y).type(torch.float32)

    def __len__(self) -> int:
        """
        Returns:
            int: n_samples (length of supplied `X`)
        """

        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, 'torch.tensor']:
        """
        Args:
            idx (int): index to query

        Returns:
            Dict[str, torch.tensor]: {'X': torch.tensor, 'y': torch.tensor}
                for given index
        """

        return {
            'X': self.X[idx],
            'y': self.y[idx]
        }


def train_model(model: 'SoftmaxMLP', dataset: 'ReactorDataset',
                epochs: int = 100, batch_size: int = 1, verbose: int = 0,
                valid_size: float = 0.33, patience: int = 32,
                **kwargs) -> Tuple['SoftmaxMLP', List[float]]:
    """
    Trains a model using supplied data

    Args:
        model (SoftmaxMLP): model to train
        dataset (ReactorDataset): dataset used for training
        epochs (int, optional): number of training epochs (iterations)
            (default: 100)
        batch_size (int, optional): size of each training batch (default: 1)
        verbose (int, optional): if >0, prints loss every `this` epochs
            (default: 0, no printing)
        valid_size (float, default 0.2): proportion of the training data to be
            used as each epoch's validation subset; validation subsets used to
            terminate training (valid loss no longer decreasing); validation
            subset chosen at random each epoch
        patience (int, default 16): how many epochs to wait for a lower
            validation loss; if not found, terminates training, restores best
            (lowest valid loss) model paramters
        **kwargs (optional): additional arguments to be passed to
            torch.optim.Adam()

    Returns:
        Tuple[SoftmaxMLP, List[float]]: (trained model, training losses)
    """

    ds_train, ds_valid = train_test_split(
        dataset, test_size=valid_size, random_state=None
    )
    dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.layers.parameters(), **kwargs)
    loss = nn.MSELoss()

    model.train()
    train_losses = []
    valid_losses = []
    _best_params = model.state_dict()
    _best_v_loss = 1e10
    _n_since_best = 0

    for epoch in range(epochs):

        ds_train, ds_valid = train_test_split(
            dataset, test_size=valid_size, random_state=None
        )
        dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dataloader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

        train_loss = 0.0

        for batch in dataloader_train:

            opt.zero_grad()
            pred = model(batch['X'])
            target = batch['y']
            _loss_val = loss(pred, target)
            _loss_val.backward()
            opt.step()
            train_loss += _loss_val.detach().item()

        train_loss /= len(dataloader_train.dataset)
        train_losses.append(train_loss)

        valid_loss = 0.0

        for batch in dataloader_valid:

            pred = model(batch['X'])
            target = batch['y']
            valid_loss += loss(pred, target).detach().item()

        valid_loss /= len(dataloader_valid.dataset)
        valid_losses.append(valid_loss)

        if valid_loss < _best_v_loss:
            _best_v_loss = valid_loss
            _best_params = model.state_dict()
            _n_since_best = 0
        elif _n_since_best > patience:
            model.load_state_dict(_best_params)
            break
        else:
            _n_since_best += 1

        if verbose > 0 and epoch % verbose == 0:
            print(f'Epoch: {epoch} | Training loss: {train_loss} ' +
                  '| Validation loss: {valid_loss}')

    model.eval()
    return (model, train_losses)
