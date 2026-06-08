#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_generalization_final.py - 模型泛化能力评估

功能：
  - 用与训练相同的流程加载数据和特征
  - 评估模型在标准/不标准信号上的泛化能力
  - 按异常类型分析检测能力
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import load

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import data_loader
from src.preprocessing import preprocess
from src.event_detection import detect_events
from src.features import extract_cycle_features

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LABELS_FILE = os.path.join(ROOT_DIR, "data", "labels_v2.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 2000.0

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_all_data():
    """加载全部数据（与训练脚本相同）"""
    labels_df = pd.read_csv(LABELS_FILE)
    label_map = {}
    for _, row in labels_df.iterrows():
        key = (row["filename"], int(row["cycle_id"]))
        label_map[key] = row["quality_label"]

    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"  标签文件: {len(labels_df)} 行")
    print(f"  加载 {len(files)} 个文件...")

    all_rows = []
    matched = 0
    unmatched = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            raw, fs = data_loader(fpath)
            filtered, _ = preprocess(raw, fs)
            result = detect_events(filtered, fs)
        except Exception:
            continue

        for cid, (s, e, pk) in enumerate(result["cycles"], start=1):
            seg1 = filtered[s:e + 1, 0]
            seg2 = filtered[s:e + 1, 1]
            feat = extract_cycle_features(seg1, seg2, fs)
            feat["filename"] = fname
            feat["cycle_id"] = cid

            key = (fname, cid)
            if key in label_map:
                feat["quality_label"] = label_map[key]
                matched += 1
            else:
                feat["quality_label"] = None
                unmatched += 1

            all_rows.append(feat)

    df = pd.DataFrame(all_rows)
    print(f"  总样本数: {len(df)}, 标签匹配: {matched}, 未匹配: {unmatched}")

    df = df[df["quality_label"].notna()].copy()
    print(f"  有效样本数: {len(df)}")
    print(f"  质量分布: {df['quality_label'].value_counts().to_dict()}")

    return df


def main():
    print("=" * 70)
    print("  模型泛化能力评估 - 标准 vs 不标准信号")
    print("=" * 70)

    # 加载模型
    print("\n[1/4] 加载模型...")
    model = load(os.path.join(MODEL_DIR, "quality_model.joblib"))
    le = load(os.path.join(MODEL_DIR, "label_encoder_quality.joblib"))

    print(f"  模型: RandomForestClassifier (Pipeline)")
    print(f"  类别: {le.classes_}")

    # 加载数据
    print("\n[2/4] 加载数据...")
    df = load_all_data()

    # 提取特征（与训练脚本相同）
    print("\n[3/4] 提取特征...")
    exclude_cols = ["filename", "cycle_id", "quality_label",
                    "start_idx", "end_idx", "start_time", "end_time"]
    all_feature_names = [c for c in df.columns if c not in exclude_cols]

    print(f"  所有可用特征: {len(all_feature_names)}")

    # 加载模型期望的特征
    feature_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    if os.path.exists(feature_path):
        feature_names = load(feature_path)
        print(f"  模型期望特征: {len(feature_names)}")
        print(f"  特征列表: {feature_names}")
    else:
        print(f"  错误: 特征列表文件不存在")
        sys.exit(1)

    # 只使用模型期望的特征
    X = df[feature_names].values
    y_true = le.transform(df["quality_label"])

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y_true.shape}")

    # 预测
    print("\n[4/4] 进行预测和评估...")
    y_pred = model.predict(X)

    # 评估指标
    print("\n" + "=" * 70)
    print("  评估结果")
    print("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n【整体性能】")
    print(f"  准确率: {acc:.4f}")
    print(f"  精确率: {prec:.4f}")
    print(f"  召回率: {rec:.4f}")
    print(f"  F1 (macro): {f1:.4f}")

    # 标准信号评估
    ok_mask = y_true == le.transform(["标准"])[0]
    if ok_mask.sum() > 0:
        ok_correct = (y_pred[ok_mask] == le.transform(["标准"])[0]).sum()
        print(f"\n【标准信号】")
        print(f"  样本数: {ok_mask.sum()}")
        print(f"  正确识别: {ok_correct}")
        print(f"  识别率: {ok_correct / ok_mask.sum():.4f}")
        print(f"  误判为不标准: {ok_mask.sum() - ok_correct}")

    # 不标准信号评估
    ng_mask = y_true == le.transform(["不标准"])[0]
    if ng_mask.sum() > 0:
        ng_correct = (y_pred[ng_mask] == le.transform(["不标准"])[0]).sum()
        print(f"\n【不标准信号】")
        print(f"  样本数: {ng_mask.sum()}")
        print(f"  正确识别: {ng_correct}")
        print(f"  识别率: {ng_correct / ng_mask.sum():.4f}")
        print(f"  漏检为标准: {ng_mask.sum() - ng_correct}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n【混淆矩阵】")
    print(f"                预测")
    print(f"                不标准  标准")
    print(f"  真实 不标准    {cm[0, 0]:4d}    {cm[0, 1]:4d}")
    print(f"      标准      {cm[1, 0]:4d}    {cm[1, 1]:4d}")

    # 分类报告
    print(f"\n【详细分类报告】")
    report = classification_report(y_true, y_pred, target_names=le.classes_,
                                   digits=4)
    print(report)

    # 保存结果
    results_file = os.path.join(OUTPUT_DIR, "generalization_final_report.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  模型泛化能力评估报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"标准信号: {ok_mask.sum()}\n")
        f.write(f"不标准信号: {ng_mask.sum()}\n\n")
        f.write(f"整体准确率: {acc:.4f}\n")
        f.write(f"整体F1 (macro): {f1:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(report)

    print(f"\n结果已保存到: {results_file}")

    # 生成可视化
    cm_fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, fontsize=12)
    ax.set_yticklabels(le.classes_, fontsize=12)
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title("混淆矩阵 - 模型泛化能力", fontsize=13)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                   fontsize=14, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.tight_layout()
    cm_png = os.path.join(OUTPUT_DIR, "generalization_confusion_matrix.png")
    cm_fig.savefig(cm_png, dpi=150, bbox_inches="tight")
    plt.close(cm_fig)
    print(f"混淆矩阵已保存到: {cm_png}")

    print("\n" + "=" * 70)
    print("  评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
