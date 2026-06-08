#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate confusion matrices for non-standard signals by abnormal type
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    print("  Non-Standard Signal Confusion Matrix Analysis")
    print("=" * 70)

    # Load labels with abnormal types
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

    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"Found {len(files)} files, {len(labels_df)} labels")

    # Collect all non-standard signals with their predictions
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

        # Use predict_quality function
        pred_result = predict_quality(filtered, fs, cycles)

        # Extract predictions for non-standard samples only
        for i, cycle_pred in enumerate(pred_result["cycle_results"]):
            cid = i + 1
            pred_label = cycle_pred["quality"]

            # Get true label and abnormal type
            key = (fname, cid)
            true_label = label_map.get(key)
            abnormal_type = abnormal_map.get(key)

            # Only keep non-standard signals
            if true_label == "不标准" and abnormal_type is not None:
                ng_samples.append({
                    "file": fname,
                    "cycle": cid,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "abnormal_type": abnormal_type
                })

    print(f"\nTotal non-standard signals: {len(ng_samples)}")

    if len(ng_samples) == 0:
        print("No non-standard signals found!")
        return

    ng_df = pd.DataFrame(ng_samples)
    print(f"Abnormal type distribution:")
    print(ng_df["abnormal_type"].value_counts())

    # Overall confusion matrix for non-standard signals
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    y_true = le.transform(ng_df["true_label"])  # All are "不标准" (0)
    y_pred = le.transform(ng_df["pred_label"])

    cm_overall = confusion_matrix(y_true, y_pred)

    # Visualization 1: Overall non-standard confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_overall, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Std", "Standard"], fontsize=12)
    ax.set_yticklabels(["Non-Std", "Standard"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix - All Non-Standard Signals", fontsize=13)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_overall[i, j]), ha="center", va="center",
                   fontsize=14, color="white" if cm_overall[i, j] > cm_overall.max() * 0.5 else "black")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "non_standard_confusion_matrix_overall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Saved: non_standard_confusion_matrix_overall.png")

    # Visualization 2: Confusion matrices by abnormal type
    abnormal_types = ng_df["abnormal_type"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    results_by_type = []

    for idx, atype in enumerate(sorted(abnormal_types)):
        mask = ng_df["abnormal_type"] == atype
        subset = ng_df[mask]

        y_true_sub = le.transform(subset["true_label"])
        y_pred_sub = le.transform(subset["pred_label"])

        cm = confusion_matrix(y_true_sub, y_pred_sub)

        # Calculate metrics
        correct = (y_pred_sub == 0).sum()  # Correctly predicted as non-standard
        total = len(y_pred_sub)
        detection_rate = correct / total if total > 0 else 0

        results_by_type.append({
            "abnormal_type": atype,
            "count": total,
            "correct": correct,
            "missed": total - correct,
            "detection_rate": detection_rate
        })

        # Plot
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 20))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Std", "Std"], fontsize=10)
        ax.set_yticklabels(["Non-Std", "Std"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(f"{atype}\n(n={total}, rate={detection_rate:.1%})", fontsize=11)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=12, color="white" if cm[i, j] > 10 else "black")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "non_standard_confusion_matrix_by_type.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: non_standard_confusion_matrix_by_type.png")

    # Print summary
    print("\n" + "=" * 70)
    print("  Non-Standard Signal Detection Summary")
    print("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    correct_overall = (y_pred == 0).sum()
    total_overall = len(y_pred)

    print(f"\nOverall Non-Standard Signals (n={total_overall}):")
    print(f"  Correctly identified as non-standard: {correct_overall}/{total_overall}")
    print(f"  Detection rate: {correct_overall/total_overall:.2%}")
    print(f"  Incorrectly classified as standard: {total_overall - correct_overall}/{total_overall}")
    print(f"  Recall (sensitivity): {rec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Non-Std  Std")
    print(f"  True Non-Std    {cm_overall[0, 0]:4d}   {cm_overall[0, 1]:4d}")
    print(f"      Std         {cm_overall[1, 0]:4d}   {cm_overall[1, 1]:4d}")

    # By type
    print(f"\n" + "=" * 70)
    print("  By Abnormal Type:")
    print("=" * 70)

    results_df = pd.DataFrame(results_by_type)
    print(results_df.to_string(index=False))

    # Save results
    results_file = os.path.join(OUTPUT_DIR, "non_standard_confusion_analysis.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Non-Standard Signal Confusion Matrix Analysis\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total Non-Standard Signals: {total_overall}\n")
        f.write(f"Correctly Identified: {correct_overall}\n")
        f.write(f"Detection Rate: {correct_overall/total_overall:.2%}\n\n")

        f.write("Overall Confusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"                Non-Std  Std\n")
        f.write(f"  True Non-Std    {cm_overall[0, 0]:4d}   {cm_overall[0, 1]:4d}\n")
        f.write(f"      Std         {cm_overall[1, 0]:4d}   {cm_overall[1, 1]:4d}\n\n")

        f.write("By Abnormal Type:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation:\n")
        f.write("- Detection Rate: Percentage of non-standard signals correctly identified\n")
        f.write("- Missed: Number of non-standard signals incorrectly classified as standard\n")

    print(f"\n[OK] Saved: non_standard_confusion_analysis.txt")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
