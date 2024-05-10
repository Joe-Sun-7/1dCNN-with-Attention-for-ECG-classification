import matplotlib.pyplot as plt
import matplotlib
import pywt
import numpy as np
import os
import numpy as np
import pandas as pd
from scipy.signal import medfilt
import scipy.io as io
from scipy import signal
from scipy.fft import fft, ifft


def normalize(data):
    data = data.astype('float')
    mx = np.max(data, axis=0).astype(np.float64)
    mn = np.min(data, axis=0).astype(np.float64)
    # Workaround to solve the problem of ZeroDivisionError
    return np.true_divide(data - mn, mx - mn, out=np.zeros_like(data - mn), where=(mx - mn)!=0)

def midnum(ecg1):
    ecg1=normalize(ecg1)    # 归一化处理
    t1=int(0.2*500)
    t2=int(0.6*500)
    ecg2=medfilt(ecg1,t1+1)
    ecg3=medfilt(ecg2,t2+1)     # 分别用200ms和600ms的中值滤波得到基线
    ecg4=ecg1-ecg3  # 得到基线滤波的结果
    plt.figure(figsize=(20, 4))
    plt.plot(ecg1)  # 输出原图像
    plt.show()
    plt.figure(figsize=(20, 4))
    plt.plot(ecg3)  # 输出基线轮廓
    plt.show()
    plt.figure(figsize=(20, 4))
    plt.plot(ecg4)  # 基线滤波结果
    plt.show()


def wavelet_denoise(data):  # data: 1d numpy
    # wavelet decomposition
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # denoise using soft threshold
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # get the denoised signal by inverse wavelet transform
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def median_denoise(signal, window_size=3):
    filtered_signal = np.zeros_like(signal)
    half_window = window_size // 2
    for i in range(half_window, len(signal) - half_window):
        window = signal[i - half_window: i + half_window + 1]
        filtered_signal[i] = np.median(window)
    return filtered_signal

def fft_denoise(ecg_signal):
    # 傅里叶变换
    ecg_fft = fft(ecg_signal)

    # 频率范围
    sampling_rate = 2000  # 假设采样率为1000 Hz
    n = len(ecg_signal)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)

    # 滤波（例如，去除高频噪声）
    ecg_fft[np.abs(frequencies) > 70] = 0

    # 逆傅里叶变换
    denoised_ecg_signal = ifft(ecg_fft)
    return denoised_ecg_signal

loadData=pd.read_csv(r'./data/train_data.csv').values    # 读取数据
#print(loadData)
ecg=loadData[:, 1:206]
# print(ecg)
ecg1 = fft_denoise(ecg)
# print(ecg1)
ecg = ecg.flatten()
ecg1 = ecg1.flatten()

matplotlib.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(30, 4))
#plt.subplot(211)
plt.plot(ecg[:206])#输出原图像
plt.title('Raw Signal')
'''
plt.subplot(212)
plt.plot(ecg1[:2560])#输出原图像
plt.title('Wavelet Denoise Signal')
plt.subplots_adjust(hspace=0.5)
'''
plt.savefig('./figure/ecg_signal.png')
plt.show()

'''
fig, axs = plt.subplots(1, 2)
axs[0].plot(ecg[:2560])
axs[0].set_title('Raw Signal')
axs[1].plot(ecg1[:2560])
axs[0].set_title('Wavelet Denoise Signal')
plt.show()
'''




