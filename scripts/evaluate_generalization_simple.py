#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_generalization_simple.py - 模型泛化能力评估（简化版）

功能：
  - 按标准/不标准信号分别评估模型性能
  - 检测模型对不标准信号的识别能力
  - 按异常类型细分分析
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
from joblib import dump, load

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
    """加载全部数据（与训练脚本相同的方式）"""
    labels_df = pd.read_csv(LABELS_FILE)
    label_map = {}
    for _, row in labels_df.iterrows():
        key = (row["filename"], int(row["cycle_id"]))
        label_map[key] = row["quality_label"]

    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"Labels: {len(labels_df)} rows, {len(files)} files")

    all_rows = []
    matched = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            raw, fs = data_loader(fpath)
            filtered, _ = preprocess(raw, fs)
            result = detect_events(filtered, fs)
        except Exception as e:
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

            all_rows.append(feat)

    df = pd.DataFrame(all_rows)
    df = df[df["quality_label"].notna()].copy()

    print(f"Total: {len(df)} samples")
    print(f"Distribution: {df['quality_label'].value_counts().to_dict()}")

    return df


def main():
    print("=" * 70)
    print("  Generalization Evaluation - Standard vs Non-Standard Signals")
    print("=" * 70)

    # Load model and label encoder
    print("\nLoading model...")
    model = load(os.path.join(MODEL_DIR, "quality_model.joblib"))
    le = load(os.path.join(MODEL_DIR, "label_encoder_quality.joblib"))
    feature_names = load(os.path.join(MODEL_DIR, "feature_names.joblib"))

    print(f"  Model: RandomForestClassifier")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {le.classes_}")

    # Load data
    print("\nLoading data...")
    df = load_all_data()

    # Extract features
    print("\nExtracting features...")
    exclude_cols = ["filename", "cycle_id", "quality_label",
                    "start_idx", "end_idx", "start_time", "end_time"]
    all_cols = [c for c in df.columns if c not in exclude_cols]

    # Use only the features the model expects
    X = df[feature_names].values
    y_true = le.transform(df["quality_label"])

    # Prediction
    print("\nMaking predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

    # Evaluation
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nOverall Performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")

    # Per-class evaluation
    print(f"\nPer-Class Evaluation:")
    for i, cls in enumerate(le.classes_):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        cls_acc = accuracy_score(y_true[mask], y_pred[mask])
        cls_prec = precision_score(y_true[mask], y_pred[mask], average="micro")
        correct = (y_pred[mask] == i).sum()
        detection_rate = correct / mask.sum()

        print(f"\n  {cls} (n={mask.sum()}):")
        print(f"    Detection Rate: {detection_rate:.4f} ({correct}/{mask.sum()})")
        print(f"    Accuracy: {cls_acc:.4f}")

    # Detailed analysis: Non-standard signals
    ng_mask = y_true == le.transform(["不标准"])[0]
    if ng_mask.sum() > 0:
        ng_correct = (y_pred[ng_mask] == le.transform(["不标准"])[0]).sum()
        print(f"\n  Non-Standard Signal Detection Rate: {ng_correct}/{ng_mask.sum()} = {ng_correct/ng_mask.sum():.4f}")
        print(f"    Model correctly identified {ng_correct} non-standard signals")
        print(f"    Missed (predicted as standard): {ng_mask.sum() - ng_correct}")

    ok_mask = y_true == le.transform(["标准"])[0]
    if ok_mask.sum() > 0:
        ok_correct = (y_pred[ok_mask] == le.transform(["标准"])[0]).sum()
        print(f"\n  Standard Signal Detection Rate: {ok_correct}/{ok_mask.sum()} = {ok_correct/ok_mask.sum():.4f}")
        print(f"    Model correctly identified {ok_correct} standard signals")
        print(f"    False alarms (predicted as non-standard): {ok_mask.sum() - ok_correct}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")

    # Classification report
    print(f"\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=le.classes_)
    print(report)

    # Save results
    results_file = os.path.join(OUTPUT_DIR, "generalization_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Model Generalization Evaluation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"Overall F1 (macro): {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
