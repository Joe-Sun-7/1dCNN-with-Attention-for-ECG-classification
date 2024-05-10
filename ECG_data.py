import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_process(data_path):
    data = pd.read_csv(data_path)
    # print(data.head())
    data1 = data['heartbeat_signals'].str.split(',', expand=True)  # 将数据按‘,’拆分
    new_names = ['signals_' + str(x + 1) for x in range(205)]  # 为新生成的列取名
    data1.columns = new_names  # 重命名新生成的列名
    data1["label"] = data["label"]  # 加入标签列
    data1 = pd.DataFrame(data1, dtype=np.float64)  # 转化为数组形式

    # 把label移到最前面
    col = list(data1)
    # col.index('label')
    col.insert(0, col.pop(col.index('label')))
    data2 = data1.loc[:, col]
    del data1
    return data2


def DA_Scaling(X, sigma=0.1):
    """数据缩放，进行样本扩增"""
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise


def assign_nan(data):
    """数据处理：删去样本末尾的0，转为NaN"""
    nptmp = data.to_numpy()
    #    print(nptmp.shape)

    left_idxs = np.arange(nptmp.shape[0])
    for ii in np.arange(nptmp.shape[1])[::-1]:
        idxs = np.where(nptmp[left_idxs, ii] <= 1.e-5)[0]
        if idxs.size > 0:
            nptmp[left_idxs[idxs], ii] = np.nan
            left_idxs = left_idxs[idxs]
        #            print(f'{ii}: {left_idxs.size}, |, {left_idxs}')
        else:
            #            print(f'Finished at {ii}')
            break

    # nptmp[:, :2] = np.nan
    return pd.DataFrame(nptmp[:, :], index=data.index, columns=data.columns[:])


def data_expand(data3):
    # 对标签为1的样本进行数据增强
    idxs = data3.query('label==1').index
    datmp = data3.iloc[idxs, :].reset_index(drop=True)
    datmp['label'] = 1
    data11 = DA_Scaling(datmp, sigma=0.1)
    for i in np.arange(16):
        data11 = pd.concat((data11, datmp), axis=0).reset_index(drop=True)
    data11['label'] = 1

    # 对标签为2的样本进行数据增强
    idxs = data3.query('label==2').index
    datmp = data3.iloc[idxs, :].reset_index(drop=True)
    datmp['label'] = 2
    data22 = DA_Scaling(datmp, sigma=0.1)
    for i in np.arange(3):
        data22 = pd.concat((data22, datmp), axis=0).reset_index(drop=True)
    data22['label'] = 2

    # 对标签为3的样本进行数据增强
    idxs = data3.query('label==3').index
    datmp = data3.iloc[idxs, :].reset_index(drop=True)
    datmp['label'] = 3
    data33 = DA_Scaling(datmp, sigma=0.1)
    for i in np.arange(3):
        data33 = pd.concat((data33, datmp), axis=0).reset_index(drop=True)
    data33['label'] = 3

    data_train = pd.concat((data3, data11), axis=0).reset_index(drop=True)
    data_train = pd.concat((data_train, data22), axis=0).reset_index(drop=True)
    data_train = pd.concat((data_train, data33), axis=0).reset_index(drop=True)

    # 打乱数据，使得不同标签样本均匀混合
    data_train = data_train.sample(frac=1, random_state=2022).reset_index(drop=True)
    del idxs, datmp
    return data_train


def row_to_column(data_train):
    # 对心电特征进行行转列处理，同时为每个心电信号加入时间步特征time
    train_heartbeat_df = data_train.iloc[:, 1:].stack()     # 生成多级索引Series
    train_heartbeat_df = train_heartbeat_df.reset_index()   # 转换回DataFrame
    train_heartbeat_df.rename(columns={"level_0": "id", "level_1": "time", 0: "heartbeat_signals"}, inplace=True)
    # 时间步是原来的signal数组索引
    train_heartbeat_df["heartbeat_signals"] = train_heartbeat_df["heartbeat_signals"].astype(float)
    return train_heartbeat_df


def main():
    data_path = r'./data/others/train.csv'
    # data_path = r'./data/testA.csv'
    data2 = data_process(data_path)
    data3 = assign_nan(data2)
    #data3 = data2
    data_train = data_expand(data3)
    # print(data_train)

    data_train1 = data_train#.iloc[:100000, :]    # 从中截取5w样本
    data_train1.to_csv(r'data/train_data_nan.csv', index=False)

    '''
    data_test = data_train.iloc[-100000:-50000, :]
    data_test.to_csv(r'data/test_data_5w2.csv', index=False)
    '''
    '''
    data_random = data_train.sample(n=50000)
    data_random.to_csv(r'data/random_data.csv', index=False)
    '''
    # train_heartbeat_df = row_to_column(data_train1)
    # data_label = data_train1.iloc[:, 0]     # 数据的label（每个信号的第一行样本为label）
    # del data2, data3
    # print(train_heartbeat_df)
    # train_heartbeat_df.to_csv('train_heartbeat_df_5w.csv', index=True)


if __name__ == '__main__':
    main()
