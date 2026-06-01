#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
regenerate_labels.py — EMG 质量标签重生成脚本

功能:
    使用第3步统一的 src/ 预处理+事件检测管线，对全部 65 个 .mat 文件
    重新检测周期、计算 RMS 比值，套用生理学阈值规则判定质量，
    输出 labels_v2.csv。

流程:
    for each .mat file:
        1. data_loader 加载原始数据
        2. src.preprocessing.preprocess 统一预处理
        3. src.event_detection.detect_events Top-K 寻峰检测周期
        4. 每个周期计算 CH2_RMS / CH1_RMS 比值
        5. 套用 quality_rules 生理学阈值判定质量
        6. 从文件名解析 action_label
        7. 写入 emg/labels_v2.csv

质量判定规则 (沿用 dataset/quality_rules.py):
    - 前平举 qpj: ratio < 0.80 → 标准
    - 侧平举 cpj: ratio > 1.30 → 标准
    - 推肩   tj:  ratio < 0.85 → 标准
    - 受试者 10, 12 为对照组 (全部不标准)

使用方式:
    cd f:/Project_final
    D:/Anaconda_202410/python.exe regenerate_labels.py
"""

import os
import sys
import glob
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import data_loader
from src.preprocessing import preprocess
from src.event_detection import detect_events

# ------------------------------------------------------------
# 全局配置
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data", "labels_v2.csv")

ACTION_MAP = {"qpj": "前平举", "cpj": "侧平举", "tj": "推肩"}

# 生理学判定规则阈值 (沿用 quality_rules.py)
QUALITY_RULES = {
    "qpj": {"direction": "upper", "threshold": 0.80,
            "primary_type": "疑似中束代偿", "secondary_type": "其他不标准"},
    "cpj": {"direction": "lower", "threshold": 1.30,
            "primary_type": "疑似前束代偿", "secondary_type": "其他不标准"},
    "tj": {"direction": "upper", "threshold": 0.85,
           "primary_type": "疑似中束代偿", "secondary_type": "其他不标准"},
}

# 控制对照组
CONTROL_SUBJECTS = {"10", "12"}


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def parse_filename(fname):
    """从文件名解析受试者编号和动作代码"""
    base = fname.replace(".mat", "")
    parts = base.split("_")
    return {
        "subject_id": parts[1],
        "action_code": parts[2],
        "state": parts[3],
        "action_label": ACTION_MAP.get(parts[2], parts[2]),
    }


def classify_quality(action_code, ratio, subject_id):
    """
    基于生理学阈值判定质量与异常子类型.
    返回 (quality_label, abnormal_type)
    """
    rule = QUALITY_RULES.get(action_code)
    if rule is None:
        return "标准", ""

    is_control = subject_id in CONTROL_SUBJECTS
    ratio_bad = False
    abnormal_type = ""

    # 1. 检查比值是否偏离生理基准
    if rule["direction"] == "upper":
        if ratio > rule["threshold"]:
            ratio_bad = True
            abnormal_type = rule["primary_type"] if ratio > 1.0 else rule["secondary_type"]
    else:  # direction == "lower"
        if ratio < rule["threshold"]:
            ratio_bad = True
            abnormal_type = rule["primary_type"] if ratio < 1.0 else rule["secondary_type"]

    # 2. 整合控制组和生理判定
    if is_control:
        quality = "不标准"
        if not ratio_bad:
            abnormal_type = "其他不标准"
    elif ratio_bad:
        quality = "不标准"
    else:
        quality = "标准"
        abnormal_type = ""

    return quality, abnormal_type


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    print("=" * 64)
    print("  EMG 质量标签重生成 (基于 src/ 统一检测管线)")
    print("=" * 64)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"\n找到 {len(files)} 个 .mat 文件\n")

    if not files:
        print("错误：未在 dataset/data/ 下找到数据文件！")
        sys.exit(1)

    all_rows = []
    total_cycles = 0
    stats = {"标准": 0, "不标准": 0}

    for fpath in files:
        fname = os.path.basename(fpath)
        info = parse_filename(fname)
        print(f"[{info['subject_id']}] {fname} ({info['action_label']})", end=" ")

        try:
            raw, fs = data_loader(fpath)
            filtered, _ = preprocess(raw, fs)
            result = detect_events(filtered, fs)
        except Exception as exc:
            print(f"  !! 处理失败: {exc}")
            continue

        cycles = result["cycles"]
        n_cycles = len(cycles)
        total_cycles += n_cycles
        valid_labeled = 0

        for cid, (s, e, pk) in enumerate(cycles, start=1):
            seg1 = filtered[s:e + 1, 0]
            seg2 = filtered[s:e + 1, 1]
            rms1 = float(np.sqrt(np.mean(seg1 ** 2)))
            rms2 = float(np.sqrt(np.mean(seg2 ** 2)))
            ratio = rms2 / rms1 if rms1 > 1e-12 else float("inf")

            quality, abnormal = classify_quality(
                info["action_code"], ratio, info["subject_id"])

            dur = (e - s) / fs
            row = {
                "filename": fname,
                "cycle_id": cid,
                "start_idx": int(s),
                "end_idx": int(e),
                "start_time": round(s / fs, 2),
                "end_time": round(e / fs, 2),
                "action_label": info["action_label"],
                "quality_label": quality,
                "abnormal_type": abnormal,
            }
            all_rows.append(row)

            if quality == "标准":
                stats["标准"] += 1
            else:
                stats["不标准"] += 1
                valid_labeled += 1

        # 进度摘要
        tag = "✓" if valid_labeled > 0 else ("○" if n_cycles > 0 else "?")
        print(f"→ {n_cycles} 周期 [{quality}/{abnormal if abnormal else '-'}]")

    # 写入 CSV
    cols = ["filename", "cycle_id", "start_idx", "end_idx",
            "start_time", "end_time", "action_label",
            "quality_label", "abnormal_type"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    # 统计
    print("\n" + "=" * 64)
    print(f"  标签重生成完成！")
    print(f"  总文件数: {len(files)}")
    print(f"  总周期数: {total_cycles}")
    total = stats["标准"] + stats["不标准"]
    if total > 0:
        print(f"  标准: {stats['标准']} ({stats['标准']/total*100:.1f}%)")
        print(f"  不标准: {stats['不标准']} ({stats['不标准']/total*100:.1f}%)")
    print(f"  输出文件: {OUTPUT_FILE}")
    print("=" * 64)


if __name__ == "__main__":
    main()
