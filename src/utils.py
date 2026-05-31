# 辅助工具

import h5py
import scipy.io as sio
import numpy as np

def data_loader(filepath):
    """
    尝试从 .mat 文件中提取 data 和 Fs，自动处理 v7.3 / v7 格式。
    返回 (data, fs)，data 形状统一为 (采样点数, 通道数)。
    """
    try:
    #v7.3处理
        with h5py.File(filepath, 'r') as f:
            data = np.array(f['data'])
            fs = np.array(f['Fs']).item()
    except (OSError, KeyError):
    #v7处理
        mat = sio.loadmat(filepath)
        data = mat['data']
        fs = mat['Fs'].squeeze()

    # 统一转置逻辑：假设通道数应该远小于采样点数
    if data.ndim >= 2:
        dim0, dim1 = data.shape
        # 如果第二维远大于第一维，则是 (通道，通道数)，需要转置
        if dim0 < dim1:
            data = data.T
    return data, fs

def compute_fft(signal, fs, remove_dc=False, window = None,
                         half_spectrum=True):
    """
    计算单通道信号的 FFT 幅度谱。

    参数
    ----------
    signal : 1-D ndarray
        输入信号（一维）。
    fs : float
        采样率 (Hz)。
    remove_dc : bool
        是否去除直流分量（默认 False）。
    window : str or None
        窗函数名称，如 'hann'、'hamming' 或 None（不窗）。
    half_spectrum : bool
        是否只返回正频率部分（单边谱），默认 True。

    返回
    -------
    freqs : ndarray
        频率轴 (Hz)。
    magnitude : ndarray
        幅度谱（单位与信号一致）。
    """
    n = len(signal)

    # 去直流
    if remove_dc:
        signal = signal - np.mean(signal)

    # 加窗
    if window is not None:
        # 使用 numpy 提供的窗函数
        try:
            win = getattr(np, window)(n)
        except AttributeError:
            raise ValueError(f"不支持的窗函数: {window}")
        signal = signal * win

    # FFT
    fft_result = np.fft.fft(signal)
    # 取幅度绝对值
    magnitude = np.abs(fft_result)

    # 生成频率轴
    freqs = np.fft.fftfreq(n, 1/fs)

    if half_spectrum:
        # 保留正频率部分
        positive_idx = freqs >= 0
        freqs = freqs[positive_idx]
        magnitude = magnitude[positive_idx]
        # 单边谱除直流外幅度×2（能量守恒）
        if n % 2 == 0:
            # 偶数长度：Nyquist 频率不翻倍
            magnitude[1:-1] *= 2
        else:
            magnitude[1:] *= 2

    return freqs, magnitude

