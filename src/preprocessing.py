# -*- coding: utf-8 -*-
"""
src/preprocessing.py — 统一可配置双通道 EMG 预处理模块

功能：
    1. 幅值换算：data * (0.286 / 6.0) — 将 24 位 AD 值转换为 mV/V
    2. 去直流：减各通道均值
    3. 高通滤波：默认 20 Hz, 4 阶 Butterworth, 零相位 (filtfilt)
    4. 低通滤波：默认 450 Hz, 10 阶 Butterworth, 零相位 (filtfilt)
    5. 工频陷波：基频 + 谐波级联卷积, 零相位 (filtfilt)

数据格式：
    - 输入: (raw_data: ndarray(N, 2), fs: float)
    - 输出: (filtered_data: ndarray(N, 2), fs: float)

设计依据（illustration.md 第三节）：
    - 采样率 Fs = 2000 Hz
    - 幅值换算系数 0.286 / 6
    - 高通: 20 Hz, 4 阶 Butterworth
    - 低通: 450 Hz, 10 阶 Butterworth
    - 工频陷波: 50 Hz 基频及其 Nyquist 范围内的谐波, 级联串联
    - 全部使用 filtfilt 零相位滤波

与旧版变更点：
    - 新增去直流步骤（旧版 emg/preprocessing.py 缺失）
    - 新增参数化接口（所有滤波参数可通过关键字控制）
    - 返回格式改为 tuple(ndarray, float)（旧版返回 dict）
    - 高通/低通改为 cascaded 而非 bandpass（保持与旧版一致的独立滤波）
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def preprocess(raw_data, fs, *,
               apply_amplitude_scaling=True,
               apply_dc_removal=True,
               apply_notch=True,
               notch_freq=50.0,
               notch_bw=3.0,
               fc_high=20.0,
               fc_low=450.0,
               order_high=4,
               order_low=10):
    """
    EMG 双通道信号统一预处理

    Parameters
    ----------
    raw_data : np.ndarray, shape (n_samples, n_channels)
        原始 EMG 数据, 每列一个通道 (通常为 2 通道).
    fs : float
        采样率 (Hz), 理论值 2000 Hz.
    apply_amplitude_scaling : bool
        是否进行 24-bit AD → 模拟值 幅值换算 (×0.286/6.0). 默认 True.
    apply_dc_removal : bool
        是否去直流 (减各通道均值). 默认 True.
    apply_notch : bool
        是否进行工频陷波. 默认 True.
    notch_freq : float
        工频基频 (Hz). 默认 50 Hz.
    notch_bw : float
        陷波带宽 (Hz). 默认 3 Hz.
    fc_high : float
        高通截止频率 (Hz). 默认 20 Hz.
    fc_low : float
        低通截止频率 (Hz). 默认 450 Hz.
    order_high : int
        高通滤波器阶数. 默认 4.
    order_low : int
        低通滤波器阶数. 默认 10.

    Returns
    -------
    filtered_data : np.ndarray, shape (n_samples, n_channels)
        预处理后的信号.
    fs : float
        采样率 (透传).
    """
    data = raw_data.astype(np.float64).copy()
    n_samples, n_channels = data.shape

    # ---- 1. 幅值换算：24-bit AD → 模拟值 (mV/V) ----
    if apply_amplitude_scaling:
        data = data * (0.286 / 6.0)

    # ---- 2. 去直流：减各通道均值 ----
    if apply_dc_removal:
        data = data - np.mean(data, axis=0)

    # ---- 3. 高通滤波器 (20 Hz, 4 阶 Butterworth) ----
    nyq = fs / 2.0
    b_high, a_high = butter(order_high, fc_high / nyq, btype='high')

    # ---- 4. 低通滤波器 (450 Hz, 10 阶 Butterworth) ----
    b_low, a_low = butter(order_low, fc_low / nyq, btype='low')

    # ---- 5. 工频陷波：基频 + 谐波, 级联卷积串联 ----
    if apply_notch:
        # 计算 Nyquist 范围内的谐波数
        n_harmonics = int(np.floor((fs / 2) / notch_freq))
        b_total = np.array([1.0])
        a_total = np.array([1.0])

        for k in range(1, n_harmonics):
            f0 = k * notch_freq
            w0 = 2.0 * f0 / fs
            Q = f0 / notch_bw
            b, a = iirnotch(w0, Q)
            b_total = np.convolve(b_total, b)
            a_total = np.convolve(a_total, a)
    else:
        b_total = np.array([1.0])
        a_total = np.array([1.0])

    # ---- 逐通道滤波 (全部零相位) ----
    filtered = np.zeros_like(data)
    for ch in range(n_channels):
        sig = data[:, ch]
        # 陷波
        if apply_notch:
            sig = filtfilt(b_total, a_total, sig)
        # 高通
        sig = filtfilt(b_high, a_high, sig)
        # 低通
        sig = filtfilt(b_low, a_low, sig)
        filtered[:, ch] = sig

    return filtered, fs


def compute_rms_envelope(signal, fs, window=0.200, overlap=0.50):
    """
    计算信号的 RMS 滑动窗口包络.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        一维输入信号.
    fs : float
        采样率 (Hz).
    window : float
        RMS 窗口时长 (秒). 默认 0.200 s.
    overlap : float
        窗口重叠率, 0~1. 默认 0.50.

    Returns
    -------
    rms : np.ndarray, shape (n_samples,)
        RMS 包络 (与输入等长).
    """
    n = len(signal)
    win_samples = int(np.round(window * fs))
    if win_samples % 2 == 0:
        win_samples += 1  # 强制奇数, 保证对称
    step = max(1, int(np.round(win_samples * (1 - overlap))))

    rms = np.zeros(n)
    count = np.zeros(n)

    for start in range(0, n - win_samples + 1, step):
        seg = signal[start:start + win_samples]
        rms_val = np.sqrt(np.mean(seg ** 2))
        end = start + win_samples
        rms[start:end] += rms_val
        count[start:end] += 1

    # 避免除以零
    count[count == 0] = 1
    rms /= count
    return rms


def moving_average(signal, window_samples):
    """
    简单移动平均平滑.

    Parameters
    ----------
    signal : np.ndarray
        一维信号.
    window_samples : int
        平滑窗口点数.

    Returns
    -------
    smoothed : np.ndarray
        平滑后信号 (与输入等长).
    """
    if window_samples <= 1:
        return signal.copy()
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(signal, kernel, mode='same')
