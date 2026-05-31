# 预处理
# 注意preprocessing里传出的是字典，画图时画图函数需要传入元组
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def preprocessing(raw_emg):
    """
    处理步骤：
        1. 将 24 位 AD 值转换为微伏 (*0.286/6)
        2. 50 Hz 及其谐波的级联陷波器（零相位滤波）
        3. 10 Hz 高通滤波（零相位）
        4. 400 Hz 低通滤波（零相位）

    raw_emg : 元组(data, fs)
        原始 EMG 信号 data, 和fs : 采样率(Hz)

    Returns filtered_data : dict
        预处理后的信号，形状与输入相同 (n_channels, n_samples) 和 fs

    注意: 
        scipy 默认数据格式(n_samples, n_channels)
    """
    raw_data, fs = raw_emg

    # ---- 1. 将 24 位 AD 采样值转换为模拟值 (μV) ----
    raw_data = raw_data * (0.286 / 6)

    n_samples, n_channels = raw_data.shape

    # ---- 滤波器设计参数 ----
    fc_hp = 20          # 高通截止频率 (Hz)
    fc_lp = 450         # 低通截止频率 (Hz)
    N_hp = 4            # 高通阶数
    N_lp = 10           # 低通阶数
    fund = 50           # 基频 (Hz)
    fund_Bw = 3         # 陷波带宽 (Hz)

    # ---- 2. 级联陷波器（50 Hz 及其谐波） ----
    H_Num = int(np.floor((fs / 2) / fund))   # 到 Nyquist 频率的谐波数
    b_total = np.array([1.0])                # 分子系数初始化为 1
    a_total = np.array([1.0])                # 分母系数初始化为 1

    # k 从 1 到 H_Num-1，对应频率 k*50 Hz（MATLAB: for k=1:H_Num-1）
    for k in range(1, H_Num):
        f0 = k * fund                       # 当前陷波频率
        w0 = 2 * f0 / fs                    # 归一化频率 (0~1)
        Q = f0 / fund_Bw                    # 品质因数
        b, a = iirnotch(w0, Q)              # 二阶陷波器
        b_total = np.convolve(b_total, b)   # 串联（卷积）
        a_total = np.convolve(a_total, a)

    # ---- 3. 高通滤波器 (20 Hz) ----
    Wc_hp = 2 * fc_hp / fs
    b_high, a_high = butter(N_hp, Wc_hp, btype='high')

    # ---- 4. 低通滤波器 (450 Hz) ----
    Wc_lp = 2 * fc_lp / fs
    b_low, a_low = butter(N_lp, Wc_lp, btype='low')

    # ---- 逐通道滤波 ----
    filtered_data = np.zeros_like(raw_data)
    for ch in range(n_channels):
        emg = raw_data[:, ch]
        # 陷波
        emg_notch = filtfilt(b_total, a_total, emg)
        # 高通
        emg_hp = filtfilt(b_high, a_high, emg_notch)
        # 低通
        filtered_data[:, ch] = filtfilt(b_low, a_low, emg_hp)

    result = {'data': filtered_data, 'fs': fs}
    return result