import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import datetime
import os
from utils import plot_history_torch, plot_heat_map
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn
from utils import wavelet_denoise, fft_denoise, median_denoise

# the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

project_path = "./"
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + r"model/SE_5fold.pt"

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 4, 128)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding='same')
        # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 4, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.se1 = SELayer(4)
        # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 16, 64)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding='same')
        # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 16, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        #self.se2 = SELayer(16)
        # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 32, 32)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding='same')
        # the third pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 32, 16)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        #self.se3 = SELayer(32)
        # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 64, 16)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding='same')
        # (batch_size, 64, 8)
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
        # reshape the tensor with shape (batch_size, 200) to (batch_size, 1, 200)
        x = x.reshape(-1, 1, 128)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.se1(x)
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
    recall = []
    F1_score = []
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
            test_acc.append(acc)
            # 计算精确率、召回率和 F1-Score
            precision.append(precision_score(y, pred_result, average='weighted'))
            #print(precision)
            recall.append(recall_score(y, pred_result, average='weighted'))
            F1_score.append(f1_score(y, pred_result, average='weighted'))

            if step_index % 10 == 0:
                loop.set_postfix(loss=loss, acc=acc)

        return {
            "loss": np.mean(test_loss),
            "acc": np.mean(test_acc),
            "precision": np.mean(precision),
            "recall": np.mean(recall),
            "f1_score": np.mean(F1_score)
        }


def train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config):
    num_epochs = config['num_epochs']
    test_loss_ls = []
    test_acc_ls = []
    test_precision_ls = []
    test_recall_ls = []
    f1_score_ls = []
    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))  # 迭代器，并产生进度条
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))    # 验证集
        train_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        test_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer)
        test_metrix = test_steps(test_loop, model, criterion)

        test_loss_ls.append(test_metrix['loss'])
        test_acc_ls.append(test_metrix['acc'])
        test_precision_ls.append(test_metrix['precision'])
        test_recall_ls.append(test_metrix['recall'])
        f1_score_ls.append(test_metrix['f1_score'])

    return {'test_loss': sum(test_loss_ls)/len(test_loss_ls),
            'test_acc': sum(test_acc_ls)/len(test_loss_ls),
            'test_precision': sum(test_precision_ls) / len(test_precision_ls),
            'test_recall': sum(test_recall_ls) / len(test_recall_ls),
            'test_f1_score': sum(f1_score_ls) / len(f1_score_ls)}


def load_data(train_path, config):
    data = pd.read_csv(train_path)
    dataset = data.values
    # 小波变换
    x = dataset[:, 1:130]
    #x = wavelet_denoise(x)
    '''
    x = x.flatten()  # 展开成一维
    x = fft_denoise(x)
    x = x.reshape((-1, 128))
    '''

    X_train = x.astype(float)
    Y_train = dataset[:, 0:1]
    Y_train = [item for sublist in Y_train for item in sublist]  # 二维矩阵降为一维矩阵

    train_dataset = ECGDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    return X_train, Y_train, train_dataloader


def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    X_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)  # dim=0增加行数，竖着连接
            y_train = np.concatenate((y_train, y_part), axis=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid


def k_fold(k, model, criterion, optimizer, config, X_train, y_train):
    test_acc = []
    test_precision = []
    test_recall = []
    test_f1_score = []
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, X_train, y_train)
        train_dataset, valid_dataset = ECGDataset(X_train, y_train), ECGDataset(X_valid, y_valid)
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
        result = train_epochs(train_dataloader, valid_dataloader, model, criterion, optimizer, config)
        print(f'第{i+1}折\n')
        test_acc.append(result['test_acc'])
        test_precision.append(result['test_precision'])
        test_recall.append(result['test_recall'])
        test_f1_score.append(result['test_f1_score'])
    return test_acc, test_precision, test_recall, test_f1_score


def main():
    random_seed = 1
    torch.manual_seed(random_seed)
    config = {
        'num_epochs': 15,
        'batch_size': 128,
        'lr': 0.003,
    }

    train_path = r'./data/data128.csv'

    X_train, y_train, test_dataloader = load_data(train_path, config)

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
        acc, precision, recall, F1_score = k_fold(5, model, criterion, optimizer, config, X_train, y_train)
        # save the model
        torch.save(model.state_dict(), model_path)
        # plot the training history
        #plot_history_torch(history)
        for i in range(5):
            print(f"{i}fold")
            print("Accuracy on valid set: {:.2f}%".format(acc[i]*100))
            print("Precision on valid set: {:.2f}%".format(precision[i] * 100))
            print("Recall on valid set: {:.2f}%".format(recall[i] * 100))
            print("F1-score on valid set: {:.2f}%".format(F1_score[i] * 100))


if __name__ == '__main__':
    main()
