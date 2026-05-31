# 可视化绘图 
# plot需传入元组

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, MaxNLocator

def plot_data(emg, title, channels=2):
    """
    emg: 绘画对象元组(data, fs)
    title:图名称
    channels:通道数,默认2
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = True    # 解决负号显示为方块的问题
    plt.rcParams['mathtext.default'] = 'regular' 

    data, fs = emg
    if channels in (1, 2):
        n_ch = min(channels, data.shape[1])
        return plot_multi_channel(data, fs, range(n_ch), title)
    else:
        return plot_with_marks(data, fs, title)


def plot_multi_channel(data, fs, channels=None, title_prefix="CH"):
    """
    用单通道函数组成多通道图

    data : np.ndarray
        形状为(n_samples,) 的多维信号。
    fs : int or float
        采样率 (Hz)。
    channels: 
        被选中通道
    title_prefix : str
        图表标题。
    Returns fig : matplotlib.figure.Figure
    """
    n_channels = data.shape[0]
    # 如果调用函数时没有指定 channels 参数,就默认绘制全部通道。
    if channels is None:
        channels = list(range(n_channels))
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2*len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    
    for ax, ch in zip(axes, channels):
        plot_single_channel(data[:,ch], fs, title=f"{title_prefix}{ch+1}", ax=ax, color = ch)
    
    # 只保留最下面子图的 xlabel，其他清空
    for ax in axes[:-1]:
        ax.set_xlabel('')
    plt.tight_layout()
    return fig

def plot_single_channel(data, fs, title="", ax= None, color = 1):
    """
    绘制单个通道的 EMG 信号。
    如果 ax 为 None,创建新 Figure:否则在传入的 ax 上绘图.

    data : np.ndarray
        形状为(n_samples) 的一维信号。
    fs : int or float
        采样率 (Hz)。
    title : str
        图表标题。
    Returns fig : matplotlib.figure.Figure
    """
    signal = data
    n_samples = len(signal)
    time = np.arange(n_samples) / fs

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure
    
    if color == 0:
        ax.plot(time, signal, linewidth=0.6, color='tab:blue')
    else:
        ax.plot(time, signal, linewidth=0.6, color='tab:red')
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('幅值')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')
    _apply_y_axis_format(ax)
    plt.tight_layout()
    return fig

def plot_with_marks(data, fs, title_prefix="CH"):
    """
    绘制两个通道的 EMG 信号，并在活动段（标记=1）上用红点高亮。

    data : np.ndarray
        形状为 (n_samples, 3) 的数组，三列分别为：
        - 通道1信号
        - 通道2信号
        - 标记 (0/1)
    fs : float
        采样率 (Hz)
    title_prefix : str
        每个子图标题的前缀，如 "通道"

    Returns：fig : matplotlib.figure.Figure
    """
    # 提取列
    B = data[:, 0]          # 通道1
    T = data[:, 1]          # 通道2
    M = data[:, 2].astype(bool)   # 布尔掩码，True 表示活动

    n_samples = len(B)
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    # 通道1
    axes[0].plot(t, B, '-b', linewidth=1.5, label='信号')
    axes[0].plot(t[M], B[M], 'ro', markersize=2, label='活动段')
    axes[0].set_ylabel('幅值')
    axes[0].set_title(f'{title_prefix}1')
    axes[0].grid(True, alpha=0.3)
    _apply_y_axis_format(axes[0])   
    axes[0].legend(loc='upper right')

    # 通道2
    axes[1].plot(t, T, '-b', linewidth=1.5, label='信号')
    axes[1].plot(t[M], T[M], 'ro', markersize=2, label='活动段')
    axes[1].set_xlabel('时间 (秒)')
    axes[1].set_ylabel('幅值')
    axes[1].set_title(f'{title_prefix}2')
    axes[1].grid(True, alpha=0.3)
    _apply_y_axis_format(axes[1])   
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    return fig

from src import utils

def plot_fft_spectrum(emg, channels=2, title="幅度谱",
                      freq_range=(0,500), scale='linear', **fft_kwargs):
    """
    绘制双通道信号的 FFT 幅度谱。

    参数
    ----------
    emg : tuple (data, fs)
        数据元组，data 形状为 (n_samples, n_channels)，fs 为采样率 (Hz)。
    channels : int
        通道数，默认 2（仅绘制前两个通道）。
    title : str
        图表总标题。
    freq_range : tuple (fmin, fmax) or None
        显示的频率范围，如 (0, 500)。默认 None 显示 0 到 Nyquist。
    **fft_kwargs :
        传递给 compute_fft_spectrum 的额外参数，
        如 remove_dc=True, window='hann', half_spectrum=True。
    scale : str, 'linear' 或 'dB'
        纵轴刻度类型：
        - 'linear' : 线性幅度
        - 'dB'     : 分贝刻度 (20*log10(幅度))

    返回
    -------
    fig : matplotlib.figure.Figure
    """
    data, fs = emg

    # 只取前 channels 列
    if data.shape[1] > channels:
        data = data[:, :channels]

    n_channels = data.shape[1]

    # 中文字体（与项目其他绘图函数一致）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                       'WenQuanYi Micro Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.default'] = 'regular' 

    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2 * n_channels),
                             sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ax, ch in zip(axes, range(n_channels)):
        signal = data[:, ch]
        freqs, mag =utils.compute_fft(signal, fs, **fft_kwargs)

        if scale == 'dB':
            # 避免 log(0) 警告，将极小值替换为一个很小的数
            eps = np.finfo(float).eps
            mag = 20 * np.log10(np.maximum(mag, eps))
            ylabel = '幅度 (dB)'
        else:
            ylabel = '幅度'

        if ch == 0:
            ax.plot(freqs, mag, linewidth=0.8, color='tab:blue')
        else:
            ax.plot(freqs, mag, linewidth=0.8, color='tab:red')
        ax.set_ylabel(ylabel)
        ax.set_title(f'通道 {ch+1}')
        ax.grid(True, alpha=0.3, linestyle='--')

        if freq_range is not None:
            ax.set_xlim(freq_range)
        
        _apply_y_axis_format(ax)

    axes[-1].set_xlabel('频率 (Hz)')
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

def _apply_y_axis_format(ax, max_ticks=6, power_limits=(-2, 2)):
    """
    统一设置纵轴格式：自动科学计数法 + 增加刻度数量
    """
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits(power_limits)          # 超过10^2或小于10^-2时启用科学计数
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_locator(MaxNLocator(max_ticks))
    ax.xaxis.set_major_locator(MaxNLocator(2*max_ticks))