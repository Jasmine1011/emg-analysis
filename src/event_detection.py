# -*- coding: utf-8 -*-
"""
src/event_detection.py — 两阶段动作事件检测模块 (v2.0)

功能：
    阶段1: 活动/静息分割 — 自适应阈值检测活动爆发段
    阶段2: 周期分割 — 基线阈值交叉法确定每个周期的激活→放松边界

算法流程：
    1. 双通道 RMS 包络 → 归一化 → 融合
    2. 平滑融合包络
    3. 自动估计静息基线 (下分位数)
    4. 自适应阈值标记活动段 (baseline + N*sigma)
    5. 合并相邻活动段
    6. 在每个活动段内：阈值交叉法检测周期边界
    7. 物理时长过滤
    8. 输出 (含三级标记)

标记体系 (segment_data 第三列):
    0 = 静息段 (rest)
    1 = 活动段 (active, 但未划分为完整周期)
    2 = 周期段 (cycle, 从激活到放松的完整动作)

设计依据：
    - illustration.md 第四节: RMS包络+自适应阈值+最小周期时长约束+峰/谷边界修正
    - 用户要求: 排除静息段，只标记从激活到运动到放松的部分
"""

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from .preprocessing import compute_rms_envelope, moving_average


# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------
def detect_events(filtered_data, fs, *,
                  rms_window=0.200,
                  overlap=0.50,
                  smooth_window=0.300,
                  min_duration=1.5,
                  max_duration=8.0,
                  min_rest=0.3,
                  target_k=5,
                  min_peak_dist=2.0,
                  prominence_factor=0.10,
                  # 内部参数（不暴露给用户）
                  activity_sigma=3.0,
                  baseline_pct=20):
    """
    两阶段动作事件检测入口.

    Parameters
    ----------
    filtered_data : np.ndarray, shape (n_samples, 2)
    fs : float
    rms_window : float, 默认 0.200 s
    overlap : float, 默认 0.50
    smooth_window : float, 默认 0.300 s
    min_duration : float, 最短周期 (s), 默认 1.5
    max_duration : float, 最长周期 (s), 默认 8.0
    min_rest : float, 最小静息间隔 (s), 默认 0.3
    target_k : int, 预期周期数, 默认 5
    min_peak_dist : float, 峰最小间距 (s), 默认 2.0
    prominence_factor : float, 默认 0.10
    activity_sigma : float, 活动阈值 sigma 倍数 (内部), 默认 3.0
    baseline_pct : float, 基线估计分位数 (内部), 默认 20

    Returns
    -------
    result : dict
    """
    n_samples = filtered_data.shape[0]
    ch1 = filtered_data[:, 0]
    ch2 = filtered_data[:, 1]

    # ---- 1. 计算双通道 RMS 包络 ----
    env1 = compute_rms_envelope(ch1, fs, window=rms_window, overlap=overlap)
    env2 = compute_rms_envelope(ch2, fs, window=rms_window, overlap=overlap)

    # ---- 2. 归一化 → 融合 ----
    env1_max = np.max(env1)
    env2_max = np.max(env2)
    env1_norm = env1 / env1_max if env1_max > 0 else env1
    env2_norm = env2 / env2_max if env2_max > 0 else env2
    envelope = 0.5 * env1_norm + 0.5 * env2_norm

    # ---- 3. 平滑 ----
    smooth_samples = int(np.round(smooth_window * fs))
    if smooth_samples > 1:
        envelope_smooth = moving_average(envelope, smooth_samples)
    else:
        envelope_smooth = envelope

    # ---- 4. 估计静息基线 ----
    baseline, sigma = _estimate_baseline(envelope_smooth, pct=baseline_pct)
    threshold = baseline + activity_sigma * sigma

    # ---- 5. 阶段1: 活动/静息分割 ----
    active_segments = _detect_active_segments(
        envelope_smooth, threshold, fs, min_rest=min_rest)

    # ---- 6. 阶段2: 每个活动段内检测周期 ----
    all_cycles = []
    cycle_id = 0

    for a_start, a_end in active_segments:
        seg_env = envelope_smooth[a_start:a_end + 1]
        seg_offset = a_start

        cycles_in_seg = _detect_cycles_in_segment(
            seg_env, seg_offset, fs, threshold,
            min_duration=min_duration,
            max_duration=max_duration,
            min_peak_dist=min_peak_dist,
            prominence_factor=prominence_factor,
            target_k=target_k,
        )

        for s, e, pk in cycles_in_seg:
            cycle_id += 1
            all_cycles.append((s, e, pk))

    # ---- 7. 如果所有活动段都无有效周期，全局回退 ----
    if not all_cycles:
        global_cycles = _detect_cycles_in_segment(
            envelope_smooth, 0, fs, threshold,
            min_duration=min_duration,
            max_duration=max_duration,
            min_peak_dist=min_peak_dist,
            prominence_factor=prominence_factor,
            target_k=target_k,
        )
        for s, e, pk in global_cycles:
            cycle_id += 1
            all_cycles.append((s, e, pk))

    # ---- 8. 生成 borders 和 segment_data ----
    borders = []
    for s, e, _ in all_cycles:
        borders.extend([s, e])

    rest_segments = _compute_rest_segments(active_segments, n_samples)
    segment_data = _make_segment_data(
        filtered_data, all_cycles, active_segments, rest_segments, fs)

    return {
        'cycles': all_cycles,
        'active_segments': active_segments,
        'rest_segments': rest_segments,
        'borders': borders,
        'segment_data': segment_data,
        'count': len(all_cycles),
        'envelope': envelope_smooth,
        'envelope_ch1': env1,
        'envelope_ch2': env2,
        'baseline': float(baseline),
        'threshold': float(threshold),
    }


# ================================================================
# 阶段1: 活动/静息分割
# ================================================================

def _estimate_baseline(envelope, pct=20):
    """
    自动估计静息基线和噪声水平.

    方法: 取包络下 pct% 分位数范围作为"静息候选"，
         其中位数 = baseline，MAD × 1.4826 = sigma.
    """
    sorted_env = np.sort(envelope)
    n_low = int(np.round(len(envelope) * pct / 100))
    n_low = max(n_low, 10)
    low_data = sorted_env[:n_low]
    baseline = float(np.median(low_data))
    sigma = float(1.4826 * np.median(np.abs(low_data - baseline)))
    return baseline, max(sigma, 1e-12)


def _detect_active_segments(envelope, threshold, fs, min_rest=0.3):
    """
    检测所有高于阈值的连续活动段，合并间距过短的相邻段.

    Returns
    -------
    segments : list of [start_idx, end_idx]
    """
    above = envelope > threshold
    n = len(envelope)

    # 找连续活动段边界
    d = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1

    if len(starts) == 0:
        return []

    segments = list(zip(starts.tolist(), ends.tolist()))

    # 合并间距过短的相邻段
    min_gap_samples = int(np.round(min_rest * fs))
    merged = [list(segments[0])]

    for i in range(1, len(segments)):
        gap = segments[i][0] - merged[-1][1] - 1
        if gap <= min_gap_samples:
            # 合并
            merged[-1][1] = segments[i][1]
        else:
            merged.append(list(segments[i]))

    return merged


def _compute_rest_segments(active_segments, total_samples):
    """根据活动段推算静息段"""
    if not active_segments:
        return [[0, total_samples - 1]]

    rests = []
    # 开头静息
    if active_segments[0][0] > 0:
        rests.append([0, active_segments[0][0] - 1])
    # 活动段之间
    for i in range(1, len(active_segments)):
        rests.append([active_segments[i - 1][1] + 1, active_segments[i][0] - 1])
    # 结尾静息
    if active_segments[-1][1] < total_samples - 1:
        rests.append([active_segments[-1][1] + 1, total_samples - 1])

    return rests


# ================================================================
# 阶段2: 活动段内周期分割 (阈值交叉法)
# ================================================================

def _detect_cycles_in_segment(seg_env, offset, fs, threshold,
                               min_duration, max_duration,
                               min_peak_dist, prominence_factor,
                               target_k):
    """
    在一个活动段内检测个体动作周期.

    方法:
        1. 找包络中所有 crossing 点（穿过阈值的时刻）
        2. 上升沿→下降沿 对 确定周期候选
        3. 用 Top-K prominence 确认主峰
        4. 每个主峰的激活起点=前一个上升沿, 放松终点=后一个下降沿
        5. 时长过滤
    """
    n_seg = len(seg_env)

    # ---- 找阈值交叉点 ----
    above = seg_env > threshold
    d = np.diff(np.concatenate(([0], above.astype(int), [0])))
    rise = np.where(d == 1)[0]           # 上升穿过阈值 (激活起点候选)
    fall = np.where(d == -1)[0] - 1      # 下降穿过阈值 (放松终点候选)

    if len(rise) == 0:
        # 整个段都在阈值之上 → 整个段是一个活动段
        rise = np.array([0])
        fall = np.array([n_seg - 1])

    # ---- 在段内找主峰 ----
    min_dist_s = int(np.round(min_peak_dist * fs))
    prom_th = prominence_factor * np.max(seg_env) if np.max(seg_env) > 0 else 0

    peaks, _ = find_peaks(seg_env, distance=max(1, min_dist_s),
                          prominence=prom_th)

    if len(peaks) == 0:
        # 无显著主峰 → 整段作为一个周期
        dur = (fall[-1] - rise[0]) / fs
        if min_duration <= dur <= max_duration:
            s_global = offset + rise[0]
            e_global = offset + fall[-1]
            pk_global = offset + int(np.argmax(seg_env))
            return [(s_global, e_global, pk_global)]
        return []

    # ---- Top-K prominence 选主峰 ----
    proms = peak_prominences(seg_env, peaks)[0]
    K = _choose_k(proms, target=target_k)
    top_idx = np.argsort(proms)[::-1][:K]
    selected_peaks = np.sort(peaks[top_idx])

    # ---- 为每个主峰确定边界: 首尾用阈值交叉, 峰间用谷底 ----
    cycles = []
    for i, pk in enumerate(selected_peaks):
        # 左边界
        if i == 0:
            # 第一个峰: 用最近的上升沿 (激活起点)
            left_candidates = rise[rise <= pk]
            left = left_candidates[-1] if len(left_candidates) > 0 else 0
        else:
            # 与前一峰之间的谷底
            seg_between = seg_env[selected_peaks[i-1]:pk + 1]
            left = selected_peaks[i-1] + int(np.argmin(seg_between))

        # 右边界
        if i == len(selected_peaks) - 1:
            # 最后一个峰: 用最近的下降沿 (放松终点)
            right_candidates = fall[fall >= pk]
            right = right_candidates[0] if len(right_candidates) > 0 else (n_seg - 1)
        else:
            # 与后一峰之间的谷底
            seg_between = seg_env[pk:selected_peaks[i+1] + 1]
            right = pk + int(np.argmin(seg_between))

        # 转为全局坐标
        s_global = offset + left
        e_global = offset + right
        pk_global = offset + pk

        dur = (e_global - s_global) / fs
        if min_duration <= dur <= max_duration:
            cycles.append((s_global, e_global, pk_global))

    return cycles


# ================================================================
# 内部辅助
# ================================================================

def _choose_k(proms, target=5):
    """自适应确定主峰数量 (保持原逻辑不变)"""
    n = len(proms)
    if n <= 3:
        return n

    ranked = np.sort(proms)[::-1]
    K = min(target, n)

    if n > target and K == target:
        if ranked[target] > 0.40 * ranked[target - 1]:
            K = target + 1

    while K > 1:
        avg_rest = np.mean(ranked[:K - 1])
        if ranked[K - 1] < 0.15 * avg_rest:
            K -= 1
        else:
            break

    return max(K, 1)


def _make_segment_data(filtered_data, cycles, active_segments, rest_segments, fs):
    """
    生成带三级标记的数据.
    标记: 0=静息, 1=活动段(非周期), 2=周期段
    """
    N = filtered_data.shape[0]
    mark = np.zeros(N, dtype=int)

    # 活动段 → 标记为 1
    for a_start, a_end in active_segments:
        a_start = max(0, a_start)
        a_end = min(N - 1, a_end)
        mark[a_start:a_end + 1] = 1

    # 周期段 → 标记为 2 (覆盖活动段标记)
    for s, e, _ in cycles:
        s = max(0, s)
        e = min(N - 1, e)
        mark[s:e + 1] = 2

    marked = np.column_stack((filtered_data, mark))
    return (marked, fs)
