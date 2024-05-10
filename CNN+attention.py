import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm
import datetime
import os
import seaborn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import wavelet_denoise, fft_denoise, median_denoise

# the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

project_path = "./"
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + r"model/SE_model.pt"

def plot_heat_map(y_test, y_pred):
    con_mat = confusion_matrix(y_test, y_pred)
    # normalize
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # plot
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('./figure/confusion_matrix_SE.png')
    plt.show()

def plot_history_torch(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history['train_acc'])
    plt.plot(history['test_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/accuracy_SE.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/loss_SE.png')
    plt.show()

class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=32):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)     # 修改为1d
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)    # 比原结构少一个维度
        return x * y.expand_as(x)


class ECALayer(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 4, 128)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=21, stride=1, padding='same')
        # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 4, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.se1 = SELayer(4)
        # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 16, 64)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=23, stride=1, padding='same')
        # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 16, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.se2 = SELayer(16)
        # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 32, 32)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding='same')
        # the third pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 32, 16)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.se3 = SELayer(32)
        # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 64, 16)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding='same')
        # the third pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 64, 8)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.se4 = SELayer(64)
        self.flatten = nn.Flatten()
        # fully connected layer, 128 nodes, output shape (batch_size, 128)
        self.fc1 = nn.Linear(64 * 8, 128)
        # Dropout layer, dropout rate = 0.3
        self.dropout = nn.Dropout(0.3)
        # fully connected layer, 5 nodes (number of classes), output shape (batch_size, 5)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # x.shape = (batch_size, 128)
        # reshape the tensor with shape (batch_size, 200) to (batch_size, 1, 128)
        x = x.reshape(-1, 1, 128)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        #x = self.se1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #x = self.se2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        #x = self.se3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.se4(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_steps(loop, model, criterion, optimizer):
    train_loss = []
    train_acc = []
    model.train()   # model: torch.nn.Module
    for step_index, (X, y) in loop:
        #print(X)
        #print(y)
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)
        pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(y, pred_result)
        train_acc.append(acc)
        if step_index % 10 == 0:
            loop.set_postfix(loss=loss, acc=acc)    # 进度条
    return {"loss": np.mean(train_loss),
            "acc": np.mean(train_acc)}


def test_steps(loop, model, criterion):
    test_loss = []
    test_acc = []
    precision = []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()

            test_loss.append(loss)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = accuracy_score(y, pred_result)
            precision.append(precision_score(y, pred_result, average='weighted'))
            #print(precision)
            test_acc.append(acc)
            if step_index % 10 == 0:
                loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(test_loss),
            "acc": np.mean(test_acc),
            'precision': np.mean(precision)}


def train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config):
    num_epochs = config['num_epochs']
    train_loss_ls = []
    train_loss_acc = []
    test_loss_ls = []
    test_loss_acc = []
    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))  # 迭代器，并产生进度条
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        train_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        test_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer)
        test_metrix = test_steps(test_loop, model, criterion)

        train_loss_ls.append(train_metrix['loss'])
        train_loss_acc.append(train_metrix['acc'])
        test_loss_ls.append(test_metrix['loss'])
        test_loss_acc.append(test_metrix['acc'])

    return {'train_loss': train_loss_ls,
            'train_acc': train_loss_acc,
            'test_loss': test_loss_ls,
            'test_acc': test_loss_acc}


def load_data(train_path, test_path, random_path, config):
    data = pd.read_csv(train_path)
    dataset = data.values
    # 小波变换
    x = dataset[:, 1:130]
    #x = x.flatten()     # 展开成一维
    x = wavelet_denoise(x)
    #x = x.reshape((-1,128))
    X_train = x.astype(float)
    Y_train = dataset[:, 0:1]
    Y_train = [item for sublist in Y_train for item in sublist]  # 二维矩阵降为一维矩阵

    data = pd.read_csv(test_path)
    dataset = data.values
    # 小波变换
    x = dataset[:, 1:130]
    #x = x.flatten()  # 展开成一维
    x = wavelet_denoise(x)
    #x = x.reshape((-1, 128))
    X_test = x.astype(float)
    Y_test = dataset[:, 0:1]
    Y_test = [item for sublist in Y_test for item in sublist]

    data = pd.read_csv(random_path)
    dataset = data.values
    # 小波变换
    x = dataset[:, 1:130]
    # = x.flatten()  # 展开成一维
    x = wavelet_denoise(x)
    #x = x.reshape((-1, 128))
    RX_test = x.astype(float)
    RY_test = dataset[:, 0:1]
    RY_test = [item for sublist in RY_test for item in sublist]

    train_dataset, test_dataset = ECGDataset(X_train, Y_train), ECGDataset(X_test, Y_test)
    random_dataset = ECGDataset(RX_test, RY_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    random_dataloader = DataLoader(random_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_dataloader, test_dataloader, RY_test, random_dataloader


def main():
    random_seed = 1
    torch.manual_seed(random_seed)
    config = {
        'num_epochs': 15,
        'batch_size': 128,
        'lr': 0.003,
    }

    train_path = r'./data/train128.csv'
    test_path = r'./data/test128.csv'
    random_path = r'./data/test128.csv'

    train_dataloader, test_dataloader, Y_test , random_dataloader = load_data(train_path, test_path,random_path,  config)

    model = Model()
    if os.path.exists(model_path):
        # import the pre-trained model if it exists
        print('Import the pre-trained model, skip the training process')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = model.to(device)
    else:
        # build the CNN model
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        # train and evaluate model
        history = train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config)
        # save the model
        torch.save(model.state_dict(), model_path)
        # plot the training history
        plot_history_torch(history)

    # predict the class of test data
    y_pred = []
    correct_predictions = 0
    total_samples = len(Y_test)
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in enumerate(random_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y_pred.extend(pred_result)

    # plot confusion matrix heat map
    plot_heat_map(Y_test, y_pred)
    for i in range(total_samples):
        if Y_test[i] == y_pred[i]:
            correct_predictions += 1
    accuracy = correct_predictions / total_samples
    print("Accuracy on test set: {:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    main()
