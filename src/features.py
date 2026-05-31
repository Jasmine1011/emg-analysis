# -*- coding: utf-8 -*-
"""
src/features.py — 周期级 EMG 特征提取模块

功能：
    A. 滑动窗口特征曲线（用于可视化）:
       - RMS 曲线 (200ms 窗)
       - MF/MDF 曲线 (500ms 窗)
       - CH2/CH1 RMS 比值曲线

    B. 周期级特征标量（用于特征表格 + 训练）:
       全部 17 个特征，每个周期双通道分别提取

    C. 批量处理 + labels.csv 标签匹配

特征清单 (共 17 个 × 2 通道 = 34 个通道级 + 7 个协同特征 = 41 个值):

    时域 (7 个):
        1. RMS  — 均方根值
        2. MAV  — 平均绝对值
        3. VAR  — 方差
        4. WL   — 波形长度
        5. ZC   — 过零率
        6. SSC  — 斜率符号变化次数
        7. IEMG — 积分肌电值

    频域 (3 个):
        8. MF   — 中位频率 (Median Frequency)
        9. MDF  — 平均功率频率 (Mean Frequency)
        10. PF  — 峰值频率 (Peak Frequency)

    双通道协同 (7 个):
        11. CH2/CH1 RMS 比值
        12. CH1 与 CH2 RMS 差值
        13. CH1 RMS 占比
        14. CH2 RMS 占比
        15. 双通道相关系数
        16. CH1 峰值时间 (归一化)
        17. CH2 峰值时间 (归一化)
        18. CH2 - CH1 激活时间差 (ms)

设计依据:
    - illustration.md 第五节: 特征清单 + 展示方式
    - 标准 EMG 特征工程文献公式
    - 与 labels.csv 周期标签对齐

注意:
    - 频域特征计算依赖 FFT, 窗口过短 (如 200ms=400样本) 频域分辨率仅 5Hz,
      因此曲线模式用 500ms 窗口, 周期级标量对整个周期做 FFT
"""

import numpy as np
import pandas as pd
import os
import csv
from .preprocessing import compute_rms_envelope


# ================================================================
# A. 滑动窗口特征曲线
# ================================================================

def compute_feature_curves(filtered_data, fs, *,
                           time_window=0.200,
                           freq_window=0.500,
                           step=0.100):
    """
    对整个信号计算滑动窗口特征曲线 (用于可视化).

    Parameters
    ----------
    filtered_data : np.ndarray, shape (n_samples, 2)
        预处理后的双通道 EMG 信号.
    fs : float
        采样率 (Hz).
    time_window : float
        时域特征窗口 (秒). 默认 0.200 s.
    freq_window : float
        频域特征窗口 (秒). 默认 0.500 s.
    step : float
        滑动步长 (秒). 默认 0.100 s.

    Returns
    -------
    curves : dict
        {
            'rms_ch1': np.ndarray,
            'rms_ch2': np.ndarray,
            'mf_ch1': np.ndarray,
            'mf_ch2': np.ndarray,
            'mdf_ch1': np.ndarray,
            'mdf_ch2': np.ndarray,
            'ratio_ch2_ch1': np.ndarray,
            'time_axis': np.ndarray,
        }
    """
    n_samples = filtered_data.shape[0]
    ch1 = filtered_data[:, 0]
    ch2 = filtered_data[:, 1]

    time_samples = int(np.round(time_window * fs))
    freq_samples = int(np.round(freq_window * fs))
    step_samples = max(1, int(np.round(step * fs)))

    # 预分配
    n_windows = (n_samples - max(time_samples, freq_samples)) // step_samples + 1
    if n_windows < 1:
        n_windows = 1

    rms_ch1 = np.zeros(n_windows)
    rms_ch2 = np.zeros(n_windows)
    mf_ch1 = np.zeros(n_windows)
    mf_ch2 = np.zeros(n_windows)
    mdf_ch1 = np.zeros(n_windows)
    mdf_ch2 = np.zeros(n_windows)
    ratio_arr = np.zeros(n_windows)
    time_axis = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step_samples
        time_axis[i] = start / fs

        # === 时域特征 (time_window) ===
        end_time = min(start + time_samples, n_samples)
        seg1_t = ch1[start:end_time]
        seg2_t = ch2[start:end_time]

        rms_ch1[i] = np.sqrt(np.mean(seg1_t ** 2))
        rms_ch2[i] = np.sqrt(np.mean(seg2_t ** 2))

        # CH2/CH1 RMS 比值 (避免除零)
        if rms_ch1[i] > 1e-12:
            ratio_arr[i] = rms_ch2[i] / rms_ch1[i]
        else:
            ratio_arr[i] = np.nan

        # === 频域特征 (freq_window) ===
        end_freq = min(start + freq_samples, n_samples)
        seg1_f = ch1[start:end_freq]
        seg2_f = ch2[start:end_freq]

        mf_ch1[i], mdf_ch1[i], _ = _compute_freq_features(seg1_f, fs)
        mf_ch2[i], mdf_ch2[i], _ = _compute_freq_features(seg2_f, fs)

    return {
        'rms_ch1': rms_ch1,
        'rms_ch2': rms_ch2,
        'mf_ch1': mf_ch1,
        'mf_ch2': mf_ch2,
        'mdf_ch1': mdf_ch1,
        'mdf_ch2': mdf_ch2,
        'ratio_ch2_ch1': ratio_arr,
        'time_axis': time_axis,
    }


# ================================================================
# B. 周期级特征标量
# ================================================================

def extract_cycle_features(ch1_segment, ch2_segment, fs):
    """
    对单个周期的双通道片段提取全部特征.

    Parameters
    ----------
    ch1_segment : np.ndarray, shape (n_samples,)
        CH1 (三角肌前束) 信号片段.
    ch2_segment : np.ndarray, shape (n_samples,)
        CH2 (三角肌中束) 信号片段.
    fs : float
        采样率 (Hz).

    Returns
    -------
    features : dict
        包含全部通道级特征和协同特征的键值对.
    """
    feat = {}

    # ---- 时域特征 (单通道) ----
    for name, seg in [('ch1', ch1_segment), ('ch2', ch2_segment)]:
        feat[f'rms_{name}'] = _rms(seg)
        feat[f'mav_{name}'] = _mav(seg)
        feat[f'var_{name}'] = _var(seg)
        feat[f'wl_{name}'] = _wl(seg)
        feat[f'zc_{name}'] = _zc(seg, fs)
        feat[f'ssc_{name}'] = _ssc(seg)
        feat[f'iemg_{name}'] = _iemg(seg)

    # ---- 频域特征 (单通道) ----
    mf1, mdf1, pf1 = _compute_freq_features(ch1_segment, fs)
    mf2, mdf2, pf2 = _compute_freq_features(ch2_segment, fs)
    feat['mf_ch1'] = mf1
    feat['mdf_ch1'] = mdf1
    feat['pf_ch1'] = pf1
    feat['mf_ch2'] = mf2
    feat['mdf_ch2'] = mdf2
    feat['pf_ch2'] = pf2

    # ---- 双通道协同特征 ----
    rms1 = feat['rms_ch1']
    rms2 = feat['rms_ch2']

    # 11. CH2/CH1 RMS 比值
    feat['ratio_rms'] = rms2 / rms1 if rms1 > 1e-12 else np.nan

    # 12. RMS 差值
    feat['diff_rms'] = rms2 - rms1

    # 13, 14. RMS 占比
    total_rms = rms1 + rms2
    if total_rms > 1e-12:
        feat['rms_ratio_ch1'] = rms1 / total_rms
        feat['rms_ratio_ch2'] = rms2 / total_rms
    else:
        feat['rms_ratio_ch1'] = np.nan
        feat['rms_ratio_ch2'] = np.nan

    # 15. 双通道 Pearson 相关系数
    if len(ch1_segment) > 1:
        feat['corr_coef'] = float(np.corrcoef(ch1_segment, ch2_segment)[0, 1])
    else:
        feat['corr_coef'] = np.nan

    # 16, 17. 各通道峰值时间 (归一化为周期时长的比例 0~1)
    n = len(ch1_segment)
    peak_idx1 = np.argmax(np.abs(ch1_segment))
    peak_idx2 = np.argmax(np.abs(ch2_segment))
    feat['peak_time_ch1'] = peak_idx1 / n if n > 0 else np.nan
    feat['peak_time_ch2'] = peak_idx2 / n if n > 0 else np.nan

    # 18. 激活时间差 (CH2 峰值 - CH1 峰值, ms)
    feat['activation_time_diff'] = ((peak_idx2 - peak_idx1) / fs) * 1000.0

    return feat


# ================================================================
# C. 批量处理 + 标签匹配
# ================================================================

def extract_all_cycles(filtered_data, fs, cycles, *,
                       filename=None,
                       labels_df=None,
                       match_tolerance=0.200):
    """
    对所有检测到的周期批量提取特征, 并匹配 labels.csv 标签.

    Parameters
    ----------
    filtered_data : np.ndarray, shape (n_samples, 2)
        预处理后的信号.
    fs : float
        采样率.
    cycles : list of tuple (start_idx, end_idx, peak_idx)
        事件检测返回的周期列表.
    filename : str or None
        当前处理的 .mat 文件名, 用于标签匹配.
    labels_df : pd.DataFrame or None
        labels.csv 内容. 若为 None 则标签留空.
    match_tolerance : float
        边界匹配容差 (秒). 默认 0.200 s.

    Returns
    -------
    features_df : pd.DataFrame
        特征表格, 每行一个周期, 列 = 特征 + 标签.
    """
    rows = []

    for cycle_id, (s, e, pk) in enumerate(cycles, start=1):
        seg1 = filtered_data[s:e + 1, 0]
        seg2 = filtered_data[s:e + 1, 1]

        feat = extract_cycle_features(seg1, seg2, fs)
        feat['cycle_id'] = cycle_id
        feat['start_idx'] = s
        feat['end_idx'] = e
        feat['start_time'] = round(s / fs, 2)
        feat['end_time'] = round(e / fs, 2)
        feat['duration'] = round((e - s) / fs, 2)

        rows.append(feat)

    df = pd.DataFrame(rows)

    # ---- 标签匹配 ----
    df['action_label'] = ''
    df['quality_label'] = ''
    df['abnormal_type'] = ''
    df['label_source'] = '未标注'

    if filename is not None and labels_df is not None:
        _match_labels(df, filename, labels_df, match_tolerance, fs)

    # 整理列顺序
    meta_cols = ['cycle_id', 'start_idx', 'end_idx', 'start_time', 'end_time',
                 'duration', 'action_label', 'quality_label', 'abnormal_type',
                 'label_source']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]

    return df


def load_labels(labels_path):
    """
    加载 labels.csv 文件.

    Parameters
    ----------
    labels_path : str
        labels.csv 文件路径.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(labels_path):
        return None
    return pd.read_csv(labels_path)


def _match_labels(df, filename, labels_df, tolerance_s, fs):
    """
    将检测到的周期边界与 labels.csv 中的标注边界进行匹配.

    匹配策略：
        以新检测的 start_idx / end_idx 为基准,
        在 labels_df 中查找同 filename 且边界偏差 < tolerance_s 的记录.
        若找到, 则拷贝其 action_label / quality_label / abnormal_type.

    Parameters
    ----------
    df : pd.DataFrame
        特征表 (in-place 修改).
    filename : str
        当前 .mat 文件名.
    labels_df : pd.DataFrame
        标签数据.
    tolerance_s : float
        边界匹配容差 (秒).
    fs : float
        采样率.
    """
    tolerance_samples = int(np.round(tolerance_s * fs))
    file_labels = labels_df[labels_df['filename'] == filename]

    if file_labels.empty:
        # 无匹配标签可用
        return

    for idx, row in df.iterrows():
        s_new = row['start_idx']
        e_new = row['end_idx']

        best_match = None
        best_dist = float('inf')

        for _, lrow in file_labels.iterrows():
            s_old = lrow['start_idx']
            e_old = lrow['end_idx']
            dist = abs(s_new - s_old) + abs(e_new - e_old)

            if dist < best_dist:
                best_dist = dist
                best_match = lrow

        # 双向容差检查
        if best_match is not None:
            s_diff = abs(s_new - best_match['start_idx'])
            e_diff = abs(e_new - best_match['end_idx'])
            if s_diff <= tolerance_samples and e_diff <= tolerance_samples:
                df.at[idx, 'action_label'] = best_match.get('action_label', '')
                df.at[idx, 'quality_label'] = best_match.get('quality_label', '')
                df.at[idx, 'abnormal_type'] = best_match.get('abnormal_type', '')
                df.at[idx, 'label_source'] = 'labels.csv 匹配'


# ================================================================
# 单特征计算函数 (时域)
# ================================================================

def _rms(segment):
    """均方根值"""
    return float(np.sqrt(np.mean(segment ** 2)))


def _mav(segment):
    """平均绝对值 (Mean Absolute Value)"""
    return float(np.mean(np.abs(segment)))


def _var(segment):
    """方差"""
    return float(np.var(segment))


def _wl(segment):
    """波形长度 (Waveform Length): 相邻采样点差值绝对值之和"""
    return float(np.sum(np.abs(np.diff(segment))))


def _zc(segment, fs, threshold=1e-6):
    """
    过零率 (Zero Crossing rate): 每秒过零次数.

    使用阈值避免基线微小波动引起误计数.
    """
    n = len(segment)
    if n < 2:
        return 0.0

    count = 0
    for i in range(n - 1):
        if (segment[i] > threshold and segment[i + 1] < -threshold) or \
           (segment[i] < -threshold and segment[i + 1] > threshold):
            count += 1

    duration = n / fs
    return count / duration if duration > 0 else 0.0


def _ssc(segment, threshold=1e-6):
    """
    斜率符号变化次数 (Slope Sign Changes).

    计数相邻差分值符号变化的次数.
    """
    if len(segment) < 3:
        return 0

    diff = np.diff(segment)
    count = 0
    for i in range(len(diff) - 1):
        if (diff[i] > threshold and diff[i + 1] < -threshold) or \
           (diff[i] < -threshold and diff[i + 1] > threshold):
            count += 1
    return count


def _iemg(segment):
    """积分肌电值 (Integrated EMG): 绝对值之和"""
    return float(np.sum(np.abs(segment)))


# ================================================================
# 单特征计算函数 (频域)
# ================================================================

def _compute_freq_features(segment, fs):
    """
    计算频域三大特征: MF, MDF, PF.

    MF (Median Frequency): 频谱功率累积到 50% 时的频率
    MDF (Mean Frequency): 频谱功率的加权平均频率
    PF (Peak Frequency): 频谱幅度最大处的频率

    Parameters
    ----------
    segment : np.ndarray
        一维信号片段.
    fs : float
        采样率.

    Returns
    -------
    mf : float
        中位频率 (Hz).
    mdf : float
        平均功率频率 (Hz).
    pf : float
        峰值频率 (Hz).
    """
    n = len(segment)
    if n < 2:
        return 0.0, 0.0, 0.0

    # FFT 计算单边功率谱
    fft_vals = np.fft.fft(segment)
    magnitude = np.abs(fft_vals)
    power = magnitude ** 2

    # 只取正频率部分 (含直流)
    half_n = n // 2 + 1
    freqs = np.fft.fftfreq(n, 1 / fs)[:half_n]
    power = power[:half_n]

    total_power = np.sum(power)
    if total_power < 1e-15:
        return 0.0, 0.0, 0.0

    # MF: 累积功率达到 50% 的频率
    cum_power = np.cumsum(power)
    half_power = total_power / 2.0
    mf_idx = np.searchsorted(cum_power, half_power)
    mf_idx = min(mf_idx, len(freqs) - 1)
    mf = float(freqs[mf_idx])

    # MDF: 功率加权平均频率
    mdf = float(np.sum(freqs * power) / total_power)

    # PF: 峰值频率 → 功率最大的频率
    pf_idx = np.argmax(power)
    pf = float(freqs[pf_idx])

    return mf, mdf, pf


# ================================================================
# 特征名称列表
# ================================================================

FEATURE_NAMES_CHANNEL = [
    'rms', 'mav', 'var', 'wl', 'zc', 'ssc', 'iemg',  # 时域 (7)
    'mf', 'mdf', 'pf',                                  # 频域 (3)
]

FEATURE_NAMES_CROSS = [
    'ratio_rms', 'diff_rms', 'rms_ratio_ch1', 'rms_ratio_ch2',
    'corr_coef', 'peak_time_ch1', 'peak_time_ch2', 'activation_time_diff',
]

# 默认勾选用于网页展示的特征
DEFAULT_SELECTED_FEATURES = [
    'rms_ch1', 'rms_ch2', 'mav_ch1', 'mav_ch2', 'wl_ch1', 'wl_ch2',
    'mf_ch1', 'mf_ch2', 'mdf_ch1', 'mdf_ch2',
    'ratio_rms', 'corr_coef',
]
