from typing import Tuple, List, Dict

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_hidden):
            self.model.append(nn.Linear(hidden_dim, hidden_dim))
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

        self.X = torch.as_tensor(X)
        self.y = torch.as_tensor(y)

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
        **kwargs (optional): additional arguments to be passed to
            torch.optim.Adam()

    Returns:
        Tuple[SoftmaxMLP, List[float]]: (trained model, training losses)
    """

    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.layers.parameters(), **kwargs)

    model.train()
    train_losses = []
    for epoch in range(epochs):

        train_loss = 0.0

        for batch in dataloader_train:

            opt.zero_grad()
            pred = model(batch['X'])
            target = batch['y']
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()
            train_loss += loss.detach().item()

        train_loss /= len(dataloader_train.dataset)
        train_losses.append(train_loss)

        if epoch % verbose == 0:
            print(f'Epoch: {epoch} | Training loss: {train_loss}')

    model.eval()
    return (model, train_losses)
