import wfdb
import pywt
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.fft import fft, ifft

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
    plt.savefig('./figure/confusion_matrix.png')
    plt.show()


def plot_history_tf(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/loss.png')
    plt.show()


def plot_history_torch(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history['train_acc'])
    plt.plot(history['test_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/loss.png')
    plt.show()


def wavelet_denoise(data):
    """小波变换降噪"""
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
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5', mode='smooth')
    return rdata


def fft_denoise(ecg_signal):
    # 傅里叶变换
    ecg_fft = fft(ecg_signal)

    # 频率范围
    sampling_rate = 1000  # 假设采样率为1000 Hz
    n = len(ecg_signal)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)

    # 滤波（例如，去除高频噪声）
    ecg_fft[np.abs(frequencies) > 50] = 0

    # 逆傅里叶变换
    denoised_ecg_signal = ifft(ecg_fft)
    return denoised_ecg_signal.real


def median_denoise(signal, window_size=3):
    filtered_signal = np.zeros_like(signal)
    half_window = window_size // 2

    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        filtered_signal[i] = np.median(signal[start:end])

    return filtered_signal