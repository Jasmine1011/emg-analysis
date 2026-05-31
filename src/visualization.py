# -*- coding: utf-8 -*-
"""
src/visualization.py — 特征曲线和包络可视化辅助函数

功能：
    1. plot_feature_curves: RMS曲线 + MF曲线 + CH2/CH1比值曲线, 标注周期边界
    2. plot_envelope_with_cycles: 在融合包络上标注波峰和谷底边界
    3. plot_cycle_table: 周期特征表格渲染 (委托 Streamlit)

设计依据:
    - illustration.md 第五节: 特征展示布局
    - 图表中文字体适配
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, MaxNLocator


# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'


def plot_envelope_with_cycles(envelope, fs, cycles,
                               envelope_ch1=None, envelope_ch2=None,
                               threshold=None,
                               title="RMS 融合包络与动作周期边界"):
    """
    绘制融合包络并标注检测到的波峰和周期边界.

    Parameters
    ----------
    envelope : np.ndarray
        融合 RMS 包络.
    fs : float
        采样率.
    cycles : list of tuple (start_idx, end_idx, peak_idx)
        检测到的周期列表.
    envelope_ch1 : np.ndarray or None
        CH1 RMS 包络 (可选, 叠加展示).
    envelope_ch2 : np.ndarray or None
        CH2 RMS 包络 (可选, 叠加展示).
    threshold : float or None
        活动阈值线 (可选, 用于展示活动/静息分界).
    title : str
        图表标题.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(envelope)
    t = np.arange(n) / fs

    fig, ax = plt.subplots(figsize=(12, 4))

    # 融合包络
    ax.plot(t, envelope, 'k-', linewidth=1.2, label='融合包络', alpha=0.9)

    # 活动阈值线
    if threshold is not None:
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=1.0,
                   alpha=0.8, label=f'活动阈值 ({threshold:.3f})')

    # 可选: 叠加单通道包络 (半透明)
    if envelope_ch1 is not None:
        env1_max = np.max(envelope_ch1)
        if env1_max > 0:
            ax.plot(t, envelope_ch1 / env1_max, 'b--', linewidth=0.6,
                    alpha=0.4, label='CH1 包络 (归一化)')
    if envelope_ch2 is not None:
        env2_max = np.max(envelope_ch2)
        if env2_max > 0:
            ax.plot(t, envelope_ch2 / env2_max, 'r--', linewidth=0.6,
                    alpha=0.4, label='CH2 包络 (归一化)')

    # 标注周期边界和波峰
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(cycles), 1)))
    for i, (s, e, pk) in enumerate(cycles):
        color = colors[i % len(colors)]
        # 周期区间填充
        ax.axvspan(t[s], t[e], alpha=0.12, color=color,
                   label=f'周期 {i+1}' if i == 0 else '')
        # 标注后续周期 (避免重复图例)
        if i > 0:
            ax.axvspan(t[s], t[e], alpha=0.12, color=color)
        # 波峰标记
        ax.plot(t[pk], envelope[pk], 'v', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=0.5)
        # 边界虚线
        ax.axvline(t[s], color=color, linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(t[e], color=color, linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('归一化幅度')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    _apply_y_axis_format(ax)
    plt.tight_layout()
    return fig


def plot_feature_curves(feature_curves, fs, cycles=None,
                        selected_features=None):
    """
    绘制特征曲线子图: RMS 曲线, MF 曲线, CH2/CH1 RMS 比值曲线.

    Parameters
    ----------
    feature_curves : dict
        compute_feature_curves() 的输出.
    fs : float
        采样率.
    cycles : list of tuple or None
        周期边界, 若提供则在图上标注.
    selected_features : list of str or None
        要绘制的特征名列表. None 表示全部默认.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if selected_features is None:
        # 默认展示: RMS_ch1, RMS_ch2, MF_ch1, MF_ch2, ratio
        selected_features = ['rms', 'mf', 'ratio']

    time_axis = feature_curves['time_axis']

    # 确定子图数量
    n_plots = 0
    if 'rms' in selected_features:
        n_plots += 1
    if 'mf' in selected_features:
        n_plots += 1
    if 'mdf' in selected_features:
        n_plots += 1
    if 'ratio' in selected_features:
        n_plots += 1

    if n_plots == 0:
        n_plots = 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots),
                              sharex=True)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # --- RMS 曲线 ---
    if 'rms' in selected_features:
        ax = axes[plot_idx]
        ax.plot(time_axis, feature_curves['rms_ch1'], 'b-', linewidth=1.0,
                label='CH1 RMS')
        ax.plot(time_axis, feature_curves['rms_ch2'], 'r-', linewidth=1.0,
                label='CH2 RMS')
        ax.set_ylabel('RMS 幅值')
        ax.set_title('RMS 包络曲线 (200ms 滑动窗)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        _apply_y_axis_format(ax)

        if cycles:
            _add_cycle_boundaries(ax, time_axis, cycles, fs)

        plot_idx += 1

    # --- MF 曲线 ---
    if 'mf' in selected_features:
        ax = axes[plot_idx]
        ax.plot(time_axis, feature_curves['mf_ch1'], 'b-', linewidth=1.0,
                label='CH1 MF')
        ax.plot(time_axis, feature_curves['mf_ch2'], 'r-', linewidth=1.0,
                label='CH2 MF')
        ax.set_ylabel('频率 (Hz)')
        ax.set_title('中位频率 (MF) 曲线 (500ms 滑动窗)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        _apply_y_axis_format(ax)

        if cycles:
            _add_cycle_boundaries(ax, time_axis, cycles, fs)

        plot_idx += 1

    # --- MDF 曲线 (可选) ---
    if 'mdf' in selected_features:
        ax = axes[plot_idx]
        ax.plot(time_axis, feature_curves['mdf_ch1'], 'b-', linewidth=1.0,
                label='CH1 MDF')
        ax.plot(time_axis, feature_curves['mdf_ch2'], 'r-', linewidth=1.0,
                label='CH2 MDF')
        ax.set_ylabel('频率 (Hz)')
        ax.set_title('平均功率频率 (MDF) 曲线 (500ms 滑动窗)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        _apply_y_axis_format(ax)

        if cycles:
            _add_cycle_boundaries(ax, time_axis, cycles, fs)

        plot_idx += 1

    # --- CH2/CH1 RMS 比值曲线 ---
    if 'ratio' in selected_features:
        ax = axes[plot_idx]
        ax.plot(time_axis, feature_curves['ratio_ch2_ch1'], 'g-', linewidth=1.0,
                label='CH2/CH1 RMS')
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_ylabel('比值')
        ax.set_title('CH2/CH1 RMS 比值曲线')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        _apply_y_axis_format(ax)

        if cycles:
            _add_cycle_boundaries(ax, time_axis, cycles, fs)

        plot_idx += 1

    axes[-1].set_xlabel('时间 (秒)')
    plt.tight_layout()
    return fig


def _add_cycle_boundaries(ax, time_axis, cycles, fs):
    """
    在特征曲线图上添加周期边界竖线.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标坐标轴.
    time_axis : np.ndarray
        曲线时间轴.
    cycles : list of tuple
        周期列表.
    fs : float
        采样率.
    """
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(cycles), 1)))
    for i, (s, e, _) in enumerate(cycles):
        t_start = s / fs
        t_end = e / fs
        color = colors[i % len(colors)]
        ax.axvline(t_start, color=color, linestyle='--', linewidth=0.6, alpha=0.5)
        ax.axvline(t_end, color=color, linestyle='--', linewidth=0.6, alpha=0.5)


def _apply_y_axis_format(ax, max_ticks=6, power_limits=(-2, 2)):
    """统一纵轴格式"""
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits(power_limits)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_locator(MaxNLocator(max_ticks))
    ax.xaxis.set_major_locator(MaxNLocator(2 * max_ticks))
