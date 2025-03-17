import numpy as np
import torch
torch.random.manual_seed(42)
import random
random.seed(42)
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing

import matplotlib.pyplot as plt

import pandas as pd





from lib.models import SimpleModel


def load_dataset():

    dt = fetch_california_housing(as_frame=True)
    df = dt['data']
    ff = dt['target']
    print(df.columns)

    print(df.head(5))
    print(ff.describe())

    x_data = df.values
    y_data = ff.values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33, random_state=42)
    norm_x = MinMaxScaler().fit(X_train)
    X_train_n = norm_x.transform(X_train)
    X_test_n = norm_x.transform(X_test)

    norm_y = MinMaxScaler().fit(y_train)
    y_train_n = norm_y.transform(y_train)
    y_test_n = norm_y.transform(y_test)


    X_train_n = torch.as_tensor(X_train_n, dtype=torch.float32)
    y_train_n = torch.as_tensor(y_train_n, dtype=torch.float32)
    X_test_n = torch.as_tensor(X_test_n, dtype=torch.float32)
    y_test_n = torch.as_tensor(y_test_n, dtype=torch.float32)

    print(f"Train: {len(X_train_n)} \nVal: {len(X_test_n)}")

    return X_train_n, X_test_n, y_train_n, y_test_n, norm_x, norm_y


def train_model(model:nn.Module, optimizer, x_data_train, x_data_val, y_data_train, y_data_val, epochs=100, **optim_args):

    opt = optimizer(params=model.parameters(), **optim_args)
    loss_fn = nn.MSELoss()


    history = {
        'train_loss': [],
        'val_loss': [],
    }

    for epoch in range(epochs):
        opt.zero_grad()

        predictions = model(x_data_train)
        loss = loss_fn(predictions, y_data_train)
        history['train_loss'].append(loss.item())

        loss.backward()
        opt.step()

        with torch.no_grad():
            val_predictions = model(x_data_val)
            val_loss = loss_fn(val_predictions, y_data_val).item()
            history['val_loss'].append(val_loss)


            if epoch % 10 == 0:
                print(f'Epoch {epoch}, loss: {loss.item()}, val_loss: {val_loss}')

    train_losses = np.array(history['train_loss'])
    val_losses = np.array(history['val_loss'])

    train_losses /= train_losses.max()
    val_losses /= val_losses.max()

    history['train_loss'] = train_losses.tolist()
    history['val_loss'] = val_losses.tolist()

    return history









if __name__ == '__main__':

    optims = [
        torch.optim.SGD,
        torch.optim.Adagrad,
        torch.optim.RMSprop,
        torch.optim.Adam,
    ]
    hists = [None for _ in range(len(optims))]

    xtrain, xtest, ytrain, ytest, normx, normy = load_dataset()

    for i, optim in enumerate(optims):
        model = SimpleModel(3, 8, [64, 128, 1])

        hists[i] = train_model(model, optims[0], xtrain, xtest, ytrain, ytest, epochs=200, lr=0.005)


    fig, ax = plt.subplots()
    for history, opt, c in zip(hists, optims, ['r', 'g', 'b', 'y']):
        name = str(opt).split('.')[-1].replace("'", '').replace('>','')
        ax.plot(history['train_loss'],f"{c}-", label=f'train{name}')
        ax.plot(history['val_loss'], f"{c}--", label=f'val{name}')
    ax.legend(ncol=2)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.show()

