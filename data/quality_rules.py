#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quality_rules.py — EMG数据质量标注与清洗脚本 (完全重构版)

功能说明：
1. 预处理模块：完全还原 project_final.pdf 路线，包含幅值换算、去直流、带通滤波与 50Hz 及其多阶谐波陷波。
2. 动作分割：基于 Top-5 波峰提取的寻峰法，不依赖基线回落。
3. 物理约束过滤：剔除时长 < 1.5s 或 > 8.0s 的无效截断动作。
4. 质量判定：控制对照组先验标记（10/12受试者为不标准），其余通过生理学 RMS 比值进行客观判定。
5. 丰富性解耦：将哑铃重量和非节拍因素仅记入备注，不参与扣减动作质量。

通道映射：
    CH1 = 三角肌前束 (Anterior Deltoid)
    CH2 = 三角肌中束 (Medial Deltoid)
"""

import os
import sys
import glob
import shutil
import csv
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert, find_peaks, peak_prominences, iirnotch

# ============================================================
# 全局配置
# ============================================================
FS = 2000                         # 采样率 (Hz)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
LABELS_FILE = os.path.join(SCRIPT_DIR, "labels.csv")

ACTION_MAP = {'qpj': '前平举', 'cpj': '侧平举', 'tj': '推肩'}
STATE_MAP  = {'1': '正常', '2': '疲劳'}

# ============================================================
# ★ 生理学判定规则阈值 ★
# ============================================================
# ratio = CH2_RMS / CH1_RMS
#   前平举(qpj): 前束主导 → ratio 应 < 0.80
#   侧平举(cpj): 中束主导 → ratio 应 > 1.30
#   推肩  (tj) : 前束略主导 → ratio 应 < 0.85 (不设下限，容纳生理学前束发达的正常变异)
QUALITY_RULES = {
    'qpj': {
        'direction': 'upper',
        'threshold': 0.80,
        'primary_type':   '疑似中束代偿',    # ratio > 1.0
        'secondary_type': '其他不标准',       # 0.80 <= ratio <= 1.0
    },
    'cpj': {
        'direction': 'lower',
        'threshold': 1.30,
        'primary_type':   '疑似前束代偿',    # ratio < 1.0
        'secondary_type': '其他不标准',       # 1.0 <= ratio <= 1.30
    },
    'tj': {
        'direction': 'upper',
        'threshold': 0.85,
        'primary_type':   '疑似中束代偿',    # ratio > 1.0
        'secondary_type': '其他不标准',       # 0.85 <= ratio <= 1.0
    },
}

# 控制对照组 — 其余受试者全部由客观生理规则判定
KNOWN_NONSTANDARD = {
    '10': '有意做不标准动作',
    '12': '有意做不标准动作',
}

# 数据丰富性说明（只做 remark 备注，不影响质量标签）
SUBJECT_NOTES = {
    '09': '不同哑铃重量: 5kg (数据丰富性)',
    '11': '不按节拍 (测试寻峰能力)',
    '13': '不同哑铃重量: 3kg (数据丰富性); 不按节拍 (测试寻峰能力)',
    '14': '不同哑铃重量: 2.5kg (数据丰富性); 不按节拍 (测试寻峰能力)',
    '15': '不同哑铃重量: 5kg (数据丰富性); 不按节拍 (测试寻峰能力)',
}

# ============================================================
# 信号处理与预处理（对齐技术路线）
# ============================================================

def preprocess_signal(data, fs=FS):
    """
    符合 project_final.pdf 技术路线的预处理模块：
    1. 幅值换算 (data * 0.286 / 6.0)
    2. 去直流 (减均值)
    3. Butterworth 带通滤波 (20–450 Hz, 4阶)
    4. 工频及谐波零相位陷波 (50Hz 及其 100, 150, ..., 400Hz 谐波)
    """
    # 1. 幅值换算
    scaled = data * (0.286 / 6.0)
    
    # 2. 去直流
    detrended = scaled - np.mean(scaled, axis=0)
    
    # 3. 20-450Hz 带通滤波
    nyq = fs / 2.0
    b_band, a_band = butter(4, [20.0 / nyq, 450.0 / nyq], btype='band')
    filtered = filtfilt(b_band, a_band, detrended, axis=0)
    
    # 4. 50Hz 及其倍频工频陷波滤波
    for freq in range(50, 450, 50):
        b_notch, a_notch = iirnotch(freq, 30.0, fs)
        filtered = filtfilt(b_notch, a_notch, filtered, axis=0)
        
    return filtered


def compute_envelope(signal, fs=FS, cutoff=6, order=2):
    """Hilbert 包络 + 低通平滑滤波 (6 Hz)"""
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, envelope)


def moving_average(signal, window_samples):
    """简单移动平均滤波器"""
    if window_samples <= 1:
        return signal.copy()
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(signal, kernel, mode='same')

# ============================================================
# 动作分割：基于 Top-K 的寻峰法
# ============================================================

def detect_cycles(ch1_filt, ch2_filt, fs=FS, target_k=5, min_peak_dist_s=2.0):
    """
    基于 Top-K 突出度排序的动作周期分割算法 (不依赖基线回落)
    """
    # 1. 计算合并包络并进行 0.25 s 移动平均平滑
    env1 = compute_envelope(ch1_filt, fs=fs)
    env2 = compute_envelope(ch2_filt, fs=fs)
    combined = env1 + env2
    smooth_win = int(0.25 * fs)
    env_smooth = moving_average(combined, smooth_win)

    n_samples = len(env_smooth)

    # 2. scipy 局部极大值寻峰
    min_dist = int(min_peak_dist_s * fs)
    all_peaks, _ = find_peaks(env_smooth, distance=min_dist)

    if len(all_peaks) == 0:
        mid = n_samples // 2
        return [(0, n_samples - 1, mid)]

    # 3. 计算峰值突出度并降序排列
    proms = peak_prominences(env_smooth, all_peaks)[0]

    # 4. 自适应提取周期峰数 K
    K = _choose_k(proms, target_k)
    top_idx = np.argsort(proms)[::-1][:K]
    selected = np.sort(all_peaks[top_idx])

    # 5. 谷底划分周期边界
    cycles = _boundaries(env_smooth, selected)
    return cycles


def _choose_k(proms, target=5):
    """自适应确定保留的周期主峰个数 K"""
    n = len(proms)
    if n <= 3:
        return n

    ranked = np.sort(proms)[::-1]
    K = min(target, n)

    # 若第 6 个峰的突出度能达到第 5 个峰的 40% 以上，说明有 6 次动作
    if n > target and K == target:
        if ranked[target] > 0.40 * ranked[target - 1]:
            K = target + 1

    # 自底向上剪枝极其微弱的伪峰
    while K > 1:
        avg_rest = np.mean(ranked[:K - 1])
        if ranked[K - 1] < 0.15 * avg_rest:
            K -= 1
        else:
            break

    return max(K, 1)


def _boundaries(envelope, peaks):
    """以波峰间的极小值（谷底）作为周期分界，首尾延伸至峰值 10% 处"""
    n = len(envelope)
    cycles = []

    for i, pk in enumerate(peaks):
        # 确定左边界
        if i == 0:
            thr = 0.10 * envelope[pk]
            left = 0
            for j in range(pk - 1, -1, -1):
                if envelope[j] < thr:
                    left = j
                    break
        else:
            seg = envelope[peaks[i - 1]:pk]
            left = peaks[i - 1] + int(np.argmin(seg))

        # 确定右边界
        if i == len(peaks) - 1:
            thr = 0.10 * envelope[pk]
            right = n - 1
            for j in range(pk + 1, n):
                if envelope[j] < thr:
                    right = j
                    break
        else:
            seg = envelope[pk:peaks[i + 1]]
            right = pk + int(np.argmin(seg))

        cycles.append((left, right, pk))

    return cycles

# ============================================================
# 质量判定与动作过滤（完全耦合技术路线）
# ============================================================

def classify_quality(action_code, ratio, subject_id):
    """
    生理学质量客观判定与异常代偿类型细分
    """
    rule = QUALITY_RULES.get(action_code)
    if rule is None:
        return '标准', '', ''

    is_known = subject_id in KNOWN_NONSTANDARD
    ratio_bad = False
    abnormal_type = ''
    remarks = []

    # 1. 检查比值偏离生理学基准的情况
    if rule['direction'] == 'upper':
        if ratio > rule['threshold']:
            ratio_bad = True
            abnormal_type = rule['primary_type'] if ratio > 1.0 else rule['secondary_type']
    else:
        if ratio < rule['threshold']:
            ratio_bad = True
            abnormal_type = rule['primary_type'] if ratio < 1.0 else rule['secondary_type']

    # 2. 整合控制组和生理判定，生成最终标签与解释
    if is_known:
        quality = '不标准'
        remarks.append(KNOWN_NONSTANDARD[subject_id])
        if ratio_bad:
            remarks.append(f'中束/前束RMS比值={ratio:.2f}')
        else:
            if not abnormal_type:
                abnormal_type = '其他不标准'
            remarks.append(f'比值正常({ratio:.2f})但受试者已知不标准')
    elif ratio_bad:
        quality = '不标准'
        remarks.append(f'中束/前束RMS比值={ratio:.2f}')
    else:
        quality = '标准'
        abnormal_type = ''

    # 3. 追加数据丰富性特征备注（不因重量和节拍扣减动作本身质量）
    note = SUBJECT_NOTES.get(subject_id)
    if note:
        remarks.append(note)

    return quality, abnormal_type, '; '.join(remarks)


def process_file(filepath, fs=FS):
    """读取文件 → 滤波去噪（耦合） → 周期分割 → 时长过滤（耦合） → 指标计算"""
    mat  = sio.loadmat(filepath)
    data = mat['data']                       # N×2

    # 1. 采用与系统方案完全耦合的预处理（幅值换算、去直流、带通、谐波陷波）
    preprocessed = preprocess_signal(data, fs=fs)
    ch1_f = preprocessed[:, 0]
    ch2_f = preprocessed[:, 1]

    # 2. 动作分割
    cycles_raw = detect_cycles(ch1_f, ch2_f, fs=fs)

    # 3. 物理时长过滤与指标计算
    results = []
    valid_cid = 1
    for cid, (s, e, pk) in enumerate(cycles_raw, start=1):
        dur = (e - s) / fs
        # 耦合 PDF 第 7 步：过滤除时长 < 1.5s 或 > 8.0s 的无效截断动作
        if 1.5 <= dur <= 8.0:
            seg1 = ch1_f[s:e + 1]
            seg2 = ch2_f[s:e + 1]
            rms1 = float(np.sqrt(np.mean(seg1 ** 2)))
            rms2 = float(np.sqrt(np.mean(seg2 ** 2)))
            ratio = rms2 / rms1 if rms1 > 0 else float('inf')
            
            results.append({
                'cycle_id':   valid_cid,
                'start_idx':  s,
                'end_idx':    e,
                'peak_idx':   pk,
                'start_time': round(s / fs, 2),
                'end_time':   round(e / fs, 2),
                'ch1_rms':    rms1,
                'ch2_rms':    rms2,
                'ratio':      ratio,
            })
            valid_cid += 1
            
    return results

# ============================================================
# 文件解析与主执行入口
# ============================================================

def parse_filename(fname):
    """从文件名解析受试者与动作"""
    base = fname.replace('.mat', '')
    parts = base.split('_')
    return {
        'subject_id':   parts[1],
        'action_code':  parts[2],
        'state':        parts[3],
        'action_label': ACTION_MAP.get(parts[2], parts[2]),
        'state_label':  STATE_MAP.get(parts[3], parts[3]),
    }


def main():
    print("=" * 64)
    print("  EMG 数据清洗与客观标注 (自顶向下完全耦合版)")
    print("  预处理链路  : 去直流 + 幅值换算 + 20-450Hz带通 + 50Hz及其谐波陷波")
    print("  周期分割法  : Top-5 波峰提取寻峰法")
    print("  物理时长过滤: 自动剔除 < 1.5s 或 > 8.0s 的残缺片段")
    print("  标注判定法  : 生理学 RMS 比值 + 控制组先验 (重量与节拍不作为扣减依据)")
    print("=" * 64)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"\n在 data 目录找到 {len(files)} 个 .mat 数据文件...\n")
    if not files:
        print("错误：未在 data/ 目录下找到数据文件，请检查！")
        sys.exit(1)

    # 清理并重新建立 dataset/ 目录
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)

    all_rows = []
    summary = {
        'total_files': len(files),
        'total_cycles': 0,
        'standard': 0,
        'nonstandard': 0,
        'types': {},
    }

    for fpath in files:
        fname = os.path.basename(fpath)
        info  = parse_filename(fname)
        print(f"[{info['subject_id']}] 处理文件: {fname}")

        try:
            cycles = process_file(fpath)
        except Exception as exc:
            print(f"  !! 处理失败: {exc}")
            continue

        print(f"  → 提取出 {len(cycles)} 个有效运动周期")

        for c in cycles:
            ql, at, rm = classify_quality(
                info['action_code'], c['ratio'], info['subject_id'])

            row = {
                'filename':      fname,
                'cycle_id':      c['cycle_id'],
                'start_idx':     c['start_idx'],
                'end_idx':       c['end_idx'],
                'start_time':    c['start_time'],
                'end_time':      c['end_time'],
                'action_label':  info['action_label'],
                'quality_label': ql,
                'abnormal_type': at,
                'remark':        rm,
            }
            all_rows.append(row)

            tag = 'OK' if ql == '标准' else 'NG'
            print(f"    周期 {c['cycle_id']:>2d}  [{c['start_idx']:>6d}-{c['end_idx']:>6d}]  "
                  f"时值=[{c['start_time']:>6.2f}-{c['end_time']:>6.2f}]s  "
                  f"ratio={c['ratio']:.3f}  [{tag}] {ql} {at}")

            # 累加统计
            summary['total_cycles'] += 1
            if ql == '标准':
                summary['standard'] += 1
            else:
                summary['nonstandard'] += 1
                if at:
                    summary['types'][at] = summary['types'].get(at, 0) + 1

        # 检查是否所有动作周期都是标准动作
        all_standard = True
        for c in cycles:
            ql, _, _ = classify_quality(
                info['action_code'], c['ratio'], info['subject_id'])
            if ql != '标准':
                all_standard = False
                break

        # 仅归档不含不标准/代偿周期的干净数据文件至 dataset/
        if len(cycles) > 0 and all_standard:
            shutil.copy2(fpath, os.path.join(DATASET_DIR, fname))
            print(f"    → 所有动作周期均标准，已归档至干净数据集目录")
        else:
            print(f"    → 包含不标准/代偿动作，不予归档至干净数据集")

    # 写入最终的 labels.csv
    cols = ['filename', 'cycle_id', 'start_idx', 'end_idx',
            'start_time', 'end_time', 'action_label',
            'quality_label', 'abnormal_type', 'remark']
    with open(LABELS_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    # 打印最终统计总结
    print("\n" + "=" * 64)
    print("  标注任务重新生成完成！")
    print(f"  总处理文件数   : {summary['total_files']}")
    print(f"  有效周期样本数 : {summary['total_cycles']}")
    print(f"  标准动作周期数 : {summary['standard']} ({summary['standard']/summary['total_cycles']*100:.1f}%)")
    print(f"  不标准动作周期数: {summary['nonstandard']} ({summary['nonstandard']/summary['total_cycles']*100:.1f}%)")
    for t, cnt in sorted(summary['types'].items()):
        print(f"    └─ {t}: {cnt} 个")
    print(f"\n  标注表保存路径: {LABELS_FILE}")
    print(f"  原始归档路径: {DATASET_DIR}")
    print("=" * 64)


if __name__ == '__main__':
    main()
