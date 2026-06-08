#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_generalization.py — 模型泛化能力评估脚本

功能:
    1. 加载已训练的质量模型
    2. 分离标准信号和不标准信号
    3. 评估模型在以下场景中的性能：
       - 全量评估
       - 标准信号上的性能（应该 100% 识别）
       - 不标准信号上的性能（检测能力）
       - 按不标准类型细分评估（前束代偿、中束代偿、其他）
    4. 生成详细的评估报告和可视化

输出:
    emg/outputs/generalization_*.csv / .png
    emg/outputs/generalization_report.txt
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

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import data_loader
from src.preprocessing import preprocess
from src.event_detection import detect_events
from src.features import extract_cycle_features

# 全局配置
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LABELS_FILE = os.path.join(ROOT_DIR, "data", "labels_v2.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 2000.0
RANDOM_STATE = 42

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_all_data_with_quality():
    """加载全部数据，包含质量标签"""
    labels_df = pd.read_csv(LABELS_FILE)
    label_map = {}
    for _, row in labels_df.iterrows():
        key = (row["filename"], int(row["cycle_id"]))
        label_map[key] = {
            "quality_label": row["quality_label"],
            "abnormal_type": row["abnormal_type"] if pd.notna(row["abnormal_type"]) else None,
        }

    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"标签文件: {len(labels_df)} 行")
    print(f"加载 {len(files)} 个文件...")

    all_rows = []
    matched = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            raw, fs = data_loader(fpath)
            filtered, _ = preprocess(raw, fs)
            result = detect_events(filtered, fs)
        except Exception as e:
            print(f"  警告: {fname} 处理失败: {e}")
            continue

        for cid, (s, e, pk) in enumerate(result["cycles"], start=1):
            seg1 = filtered[s:e + 1, 0]
            seg2 = filtered[s:e + 1, 1]
            feat = extract_cycle_features(seg1, seg2, fs)
            feat["filename"] = fname
            feat["cycle_id"] = cid

            key = (fname, cid)
            if key in label_map:
                feat["quality_label"] = label_map[key]["quality_label"]
                feat["abnormal_type"] = label_map[key]["abnormal_type"]
                matched += 1
            else:
                continue

            all_rows.append(feat)

    df = pd.DataFrame(all_rows)
    print(f"  总样本数: {len(df)}, 标签匹配: {matched}")
    print(f"  质量分布: {df['quality_label'].value_counts().to_dict()}")

    if "abnormal_type" in df.columns:
        print(f"  不标准类型分布: {df[df['quality_label']=='不标准']['abnormal_type'].value_counts().to_dict()}")

    return df


def evaluate_generalization():
    """评估模型泛化能力"""
    print("=" * 70)
    print("  Model Generalization Evaluation")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading data...")
    df = load_all_data_with_quality()

    # 2. Load model
    print("\n[2/4] Loading pre-trained model...")
    model_path = os.path.join(MODEL_DIR, "quality_model.joblib")
    le_path = os.path.join(MODEL_DIR, "label_encoder_quality.joblib")
    feature_path = os.path.join(MODEL_DIR, "feature_names.joblib")

    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    model = load(model_path)
    le = load(le_path)

    # Load feature names
    if os.path.exists(feature_path):
        feature_names = load(feature_path)
        print(f"  Loaded feature list: {len(feature_names)} features")
        print(f"  Features: {feature_names}")
    else:
        print(f"Error: Feature names file not found: {feature_path}")
        sys.exit(1)

    # Get classifier
    clf = model.named_steps.get('c') or model.named_steps.get('clf') or list(model.named_steps.values())[-1]
    print(f"  Model type: {type(clf).__name__}")

    # 3. Prediction
    print("\n[3/4] Making predictions...")
    # Check if all required features are in the data
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]

    if missing_features:
        print(f"  Warning: Missing {len(missing_features)} features: {missing_features}")

    if len(available_features) < len(feature_names):
        print(f"  Error: Not enough features. Expected {len(feature_names)}, got {len(available_features)}")
        sys.exit(1)

    X = df[feature_names].values
    y_true = le.transform(df["quality_label"])
    y_pred = model.predict(X)

    df["y_pred"] = le.inverse_transform(y_pred)
    df["y_pred_numeric"] = y_pred

    # 4. 分类评估
    print("\n[4/4] 生成评估报告...")

    results = []

    # 4.1 全量评估
    print("\n  【全量评估】")
    metrics = {
        "数据集": "全量 (All)",
        "样本数": len(df),
        "准确率": round(accuracy_score(y_true, y_pred), 4),
        "精确率": round(precision_score(y_true, y_pred, average="macro"), 4),
        "召回率": round(recall_score(y_true, y_pred, average="macro"), 4),
        "F1 (macro)": round(f1_score(y_true, y_pred, average="macro"), 4),
    }
    results.append(metrics)
    print(f"    准确率: {metrics['准确率']:.4f}")
    print(f"    F1 (macro): {metrics['F1 (macro)']:.4f}")

    # 4.2 按真实标签分类评估
    print("\n  【按真实标签分类】")
    for label in le.classes_:
        mask = df["quality_label"] == label
        if mask.sum() == 0:
            continue

        y_true_subset = y_true[mask]
        y_pred_subset = y_pred[mask]

        acc = accuracy_score(y_true_subset, y_pred_subset)
        prec = precision_score(y_true_subset, y_pred_subset, average="macro", zero_division=0)
        rec = recall_score(y_true_subset, y_pred_subset, average="macro", zero_division=0)
        f1 = f1_score(y_true_subset, y_pred_subset, average="macro", zero_division=0)

        metrics = {
            "数据集": f"{label} (n={mask.sum()})",
            "样本数": mask.sum(),
            "准确率": round(acc, 4),
            "精确率": round(prec, 4),
            "召回率": round(rec, 4),
            "F1 (macro)": round(f1, 4),
        }
        results.append(metrics)
        print(f"    {label}: 准确率={acc:.4f}, F1={f1:.4f}, 样本数={mask.sum()}")

    # 4.3 按不标准类型细分
    print("\n  【按不标准类型细分】")
    ng_df = df[df["quality_label"] == "不标准"]

    if len(ng_df) > 0:
        abnormal_types = ng_df["abnormal_type"].dropna().unique()
        print(f"    发现 {len(abnormal_types)} 种不标准类型")

        for atype in abnormal_types:
            mask = (df["quality_label"] == "不标准") & (df["abnormal_type"] == atype)
            if mask.sum() == 0:
                continue

            y_true_subset = y_true[mask]
            y_pred_subset = y_pred[mask]

            acc = accuracy_score(y_true_subset, y_pred_subset)
            prec = precision_score(y_true_subset, y_pred_subset, average="macro", zero_division=0)
            rec = recall_score(y_true_subset, y_pred_subset, average="macro", zero_division=0)
            f1 = f1_score(y_true_subset, y_pred_subset, average="macro", zero_division=0)

            # 计算模型对不标准的检出率
            correct = (y_pred_subset == le.transform(["不标准"])[0]).sum()
            detection_rate = correct / len(y_pred_subset) if len(y_pred_subset) > 0 else 0

            metrics = {
                "数据集": f"{atype} (n={mask.sum()})",
                "样本数": mask.sum(),
                "准确率": round(acc, 4),
                "精确率": round(prec, 4),
                "召回率": round(rec, 4),
                "F1 (macro)": round(f1, 4),
            }
            results.append(metrics)
            print(f"    {atype}: 准确率={acc:.4f}, 检出率={detection_rate:.4f}, 样本数={mask.sum()}")

    # 5. 生成报告
    print("\n" + "=" * 70)
    print("  评估完成，生成报告...")
    print("=" * 70)

    # 5.1 保存指标表
    df_results = pd.DataFrame(results)
    results_csv = os.path.join(OUTPUT_DIR, "generalization_metrics.csv")
    df_results.to_csv(results_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 指标表: {results_csv}")
    print(df_results.to_string(index=False))

    # 5.2 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, fontsize=12)
    ax.set_yticklabels(le.classes_, fontsize=12)
    ax.set_xlabel("预测标签", fontsize=13)
    ax.set_ylabel("真实标签", fontsize=13)
    ax.set_title("泛化能力评估 - 混淆矩阵 (全量)", fontsize=14)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                   fontsize=14, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.tight_layout()
    cm_png = os.path.join(OUTPUT_DIR, "generalization_confusion_matrix.png")
    fig.savefig(cm_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ 混淆矩阵: {cm_png}")

    # 5.3 按类型分布可视化
    if len(ng_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        atype_counts = ng_df["abnormal_type"].value_counts()
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        ax.barh(range(len(atype_counts)), atype_counts.values, color=colors[:len(atype_counts)])
        ax.set_yticks(range(len(atype_counts)))
        ax.set_yticklabels(atype_counts.index, fontsize=11)
        ax.set_xlabel("样本数", fontsize=12)
        ax.set_title("不标准信号分布 - 按异常类型", fontsize=13)
        ax.invert_yaxis()
        for i, v in enumerate(atype_counts.values):
            ax.text(v + 1, i, str(v), va='center', fontsize=11)
        plt.tight_layout()
        atype_png = os.path.join(OUTPUT_DIR, "generalization_abnormal_types.png")
        fig.savefig(atype_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ 不标准类型分布: {atype_png}")

    # 5.4 保存详细预测结果
    prediction_cols = ["filename", "cycle_id", "quality_label", "y_pred"]
    if "abnormal_type" in df.columns:
        prediction_cols.insert(3, "abnormal_type")
    prediction_df = df[prediction_cols].copy()
    prediction_csv = os.path.join(OUTPUT_DIR, "generalization_predictions.csv")
    prediction_df.to_csv(prediction_csv, index=False, encoding="utf-8-sig")
    print(f"✓ 详细预测: {prediction_csv}")

    # 5.5 生成文字报告
    report_txt = os.path.join(OUTPUT_DIR, "generalization_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  模型泛化能力评估报告\n")
        f.write("=" * 70 + "\n\n")

        f.write("【总体指标】\n")
        f.write(f"  总样本数: {len(df)}\n")
        f.write(f"  标准信号: {(df['quality_label']=='标准').sum()}\n")
        f.write(f"  不标准信号: {(df['quality_label']=='不标准').sum()}\n")
        f.write(f"  整体准确率: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"  整体 F1 (macro): {f1_score(y_true, y_pred, average='macro'):.4f}\n\n")

        f.write("【标准信号检测】\n")
        ok_mask = df["quality_label"] == "标准"
        ok_correct = (y_pred[ok_mask] == le.transform(["标准"])[0]).sum()
        f.write(f"  检测数: {ok_mask.sum()}\n")
        f.write(f"  正确检测: {ok_correct}\n")
        f.write(f"  检测率: {ok_correct / ok_mask.sum():.4f}\n")
        f.write(f"  误判为不标准: {ok_mask.sum() - ok_correct}\n\n")

        f.write("【不标准信号检测】\n")
        ng_mask = df["quality_label"] == "不标准"
        ng_correct = (y_pred[ng_mask] == le.transform(["不标准"])[0]).sum()
        f.write(f"  检测数: {ng_mask.sum()}\n")
        f.write(f"  正确检测: {ng_correct}\n")
        f.write(f"  检测率: {ng_correct / ng_mask.sum():.4f}\n")
        f.write(f"  漏检为标准: {ng_mask.sum() - ng_correct}\n\n")

        if len(ng_df) > 0:
            f.write("【不标准类型细分】\n")
            for atype in ng_df["abnormal_type"].dropna().unique():
                atype_mask = (df["quality_label"] == "不标准") & (df["abnormal_type"] == atype)
                atype_correct = (y_pred[atype_mask] == le.transform(["不标准"])[0]).sum()
                f.write(f"  {atype}:\n")
                f.write(f"    样本数: {atype_mask.sum()}\n")
                f.write(f"    检出率: {atype_correct / atype_mask.sum():.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("  分类报告\n")
        f.write("=" * 70 + "\n\n")
        report = classification_report(y_true, y_pred, target_names=le.classes_)
        f.write(report)

    print(f"✓ 详细报告: {report_txt}")
    print("\n" + "=" * 70)
    print("  评估完成！")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_generalization()
