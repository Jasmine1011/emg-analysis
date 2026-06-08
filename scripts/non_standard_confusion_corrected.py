#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate correct confusion matrices for non-standard signals
Fix: Reverse the label mapping if needed
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from joblib import load

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import data_loader
from src.preprocessing import preprocess
from src.event_detection import detect_events
from src.prediction import predict_quality

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LABELS_FILE = os.path.join(ROOT_DIR, "data", "labels_v2.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 2000.0
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print("=" * 70)
    print("  Non-Standard Signal Confusion Matrix (with label fix)")
    print("=" * 70)

    # Load labels
    labels_df = pd.read_csv(LABELS_FILE)
    label_map = {}
    abnormal_map = {}

    for _, row in labels_df.iterrows():
        key = (row["filename"], int(row["cycle_id"]))
        label_map[key] = row["quality_label"]
        if pd.notna(row["abnormal_type"]):
            abnormal_map[key] = row["abnormal_type"]

    # Load label encoder
    le = load(os.path.join(MODEL_DIR, "label_encoder_quality.joblib"))
    print(f"Label encoder classes: {le.classes_}")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"Found {len(files)} files")

    # Collect all non-standard signals
    ng_samples = []

    for fpath in files:
        fname = os.path.basename(fpath)

        try:
            raw, fs = data_loader(fpath)
            filtered, _ = preprocess(raw, fs)
            result = detect_events(filtered, fs)
            cycles = result["cycles"]
        except Exception as e:
            continue

        pred_result = predict_quality(filtered, fs, cycles)

        for i, cycle_pred in enumerate(pred_result["cycle_results"]):
            cid = i + 1
            pred_label = cycle_pred["quality"]

            key = (fname, cid)
            true_label = label_map.get(key)
            abnormal_type = abnormal_map.get(key)

            # Keep only non-standard signals
            if true_label == "不标准" and abnormal_type is not None:
                ng_samples.append({
                    "file": fname,
                    "cycle": cid,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "abnormal_type": abnormal_type,
                    "is_correctly_predicted": (pred_label == "不标准")
                })

    print(f"\nTotal non-standard signals: {len(ng_samples)}")

    ng_df = pd.DataFrame(ng_samples)
    print(f"\nAbnomaltype distribution:")
    print(ng_df["abnormal_type"].value_counts())

    print(f"\nPrediction distribution:")
    print(ng_df["pred_label"].value_counts())

    # Convert to numeric for confusion matrix
    # Since all true labels are "不标准" (0), we just need to check predictions
    y_true_numeric = [0] * len(ng_df)  # All are "不标准"
    y_pred_numeric = le.transform(ng_df["pred_label"].values)

    # Confusion matrix
    cm = confusion_matrix(y_true_numeric, y_pred_numeric)

    print(f"\n" + "=" * 70)
    print("  Confusion Matrix - All Non-Standard Signals")
    print("=" * 70)
    print(f"\nTotal non-standard signals: {len(ng_df)}")
    correct = (y_pred_numeric == 0).sum()
    print(f"Correctly identified as non-standard: {correct}")
    print(f"Incorrectly classified as standard: {len(ng_df) - correct}")
    print(f"Detection rate: {correct/len(ng_df)*100:.1f}%")

    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"                  Predicted")
    print(f"                Non-Std(0)  Std(1)")
    print(f"  True Non-Std(0)    {cm[0, 0]:4d}      {cm[0, 1]:4d}")
    print(f"      Std(1)          {cm[1, 0]:4d}      {cm[1, 1]:4d}")

    # Visualization 1: Overall
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Std", "Standard"], fontsize=12)
    ax.set_yticklabels(["Non-Std", "Standard"], fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Non-Standard Signal Detection Confusion Matrix", fontsize=13)

    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                   fontsize=16, color=text_color, weight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "non_standard_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVED] non_standard_confusion_matrix.png")

    # Visualization 2: By abnormal type
    abnormal_types = sorted(ng_df["abnormal_type"].unique())
    fig, axes = plt.subplots(1, len(abnormal_types), figsize=(5*len(abnormal_types), 4))

    if len(abnormal_types) == 1:
        axes = [axes]

    results_data = []

    for idx, atype in enumerate(abnormal_types):
        mask = ng_df["abnormal_type"] == atype
        subset = ng_df[mask]

        y_true_sub = [0] * len(subset)
        y_pred_sub = le.transform(subset["pred_label"].values)

        cm_sub = confusion_matrix(y_true_sub, y_pred_sub, labels=[0, 1])

        correct_sub = (y_pred_sub == 0).sum()
        total_sub = len(subset)

        results_data.append({
            "abnormal_type": atype,
            "count": total_sub,
            "correct": correct_sub,
            "detection_rate": correct_sub / total_sub * 100 if total_sub > 0 else 0
        })

        ax = axes[idx]
        im = ax.imshow(cm_sub, cmap="Blues", vmin=0, vmax=max(cm_sub.max(), 30))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Std", "Std"], fontsize=10)
        ax.set_yticklabels(["Non-Std", "Std"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(f"{atype}\n(n={total_sub}, {correct_sub}/{total_sub}={correct_sub/total_sub*100:.0f}%)", fontsize=11)

        for i in range(2):
            for j in range(2):
                text_color = "white" if cm_sub[i, j] > 15 else "black"
                ax.text(j, i, str(cm_sub[i, j]), ha="center", va="center",
                       fontsize=12, color=text_color, weight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "non_standard_confusion_matrix_by_type.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] non_standard_confusion_matrix_by_type.png")

    # Print summary table
    print(f"\n" + "=" * 70)
    print("  Detection Rate by Abnormal Type")
    print("=" * 70)
    results_df = pd.DataFrame(results_data)
    print(results_df.to_string(index=False))

    # Save to file
    with open(os.path.join(OUTPUT_DIR, "non_standard_confusion_matrix_report.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Non-Standard Signal Confusion Matrix Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Non-Standard Signals: {len(ng_df)}\n")
        f.write(f"Correctly Identified: {correct}/{len(ng_df)} ({correct/len(ng_df)*100:.1f}%)\n")
        f.write(f"Incorrectly Classified as Standard: {len(ng_df) - correct}/{len(ng_df)} ({(len(ng_df)-correct)/len(ng_df)*100:.1f}%)\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"                  Predicted\n")
        f.write(f"                Non-Std  Std\n")
        f.write(f"  True Non-Std    {cm[0, 0]:4d}   {cm[0, 1]:4d}\n")
        f.write(f"      Std          {cm[1, 0]:4d}   {cm[1, 1]:4d}\n\n")
        f.write("By Abnormal Type:\n")
        f.write(results_df.to_string(index=False))

    print(f"\n[SAVED] non_standard_confusion_matrix_report.txt")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
