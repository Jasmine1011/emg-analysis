#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_with_app_functions.py - 使用应用中的预测函数进行评估
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import data_loader
from src.preprocessing import preprocess
from src.event_detection import detect_events
from src.prediction import predict_quality
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from joblib import load

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
    print("  Using App Prediction Functions for Evaluation")
    print("=" * 70)

    # Load labels
    labels_df = pd.read_csv(LABELS_FILE)
    label_map = {}
    for _, row in labels_df.iterrows():
        key = (row["filename"], int(row["cycle_id"]))
        label_map[key] = row["quality_label"]

    # Load label encoder
    le = load(os.path.join(MODEL_DIR, "label_encoder_quality.joblib"))

    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"Found {len(files)} files, {len(labels_df)} labels")

    # Process each file
    all_predictions = []
    all_true_labels = []

    for fpath in files:
        fname = os.path.basename(fpath)

        try:
            raw, fs = data_loader(fpath)
            filtered, _ = preprocess(raw, fs)
            result = detect_events(filtered, fs)
            cycles = result["cycles"]
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

        # Use predict_quality function (same as in app)
        pred_result = predict_quality(filtered, fs, cycles)

        # Extract predictions
        for i, cycle_pred in enumerate(pred_result["cycle_results"]):
            cid = i + 1
            pred_label = cycle_pred["quality"]

            # Get true label
            key = (fname, cid)
            true_label = label_map.get(key)

            if true_label is not None:
                all_predictions.append(pred_label)
                all_true_labels.append(true_label)

    # Evaluate
    print(f"\nTotal predictions: {len(all_predictions)}")
    print(f"Total true labels: {len(all_true_labels)}")

    if len(all_predictions) > 0:
        y_true = le.transform(all_true_labels)
        y_pred = le.transform(all_predictions)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro")
        rec = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

        print("\n" + "=" * 70)
        print("  EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 (macro): {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Non-standard  Standard")
        print(f"  True Non-standard    {cm[0, 0]:4d}    {cm[0, 1]:4d}")
        print(f"      Standard        {cm[1, 0]:4d}    {cm[1, 1]:4d}")

        # Classification report
        report = classification_report(y_true, y_pred, target_names=le.classes_)
        print(f"\nDetailed Report:")
        print(report)

        # Save results
        results_file = os.path.join(OUTPUT_DIR, "evaluation_with_app_functions.txt")
        with open(results_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("  Model Evaluation Using App Functions\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total Samples: {len(all_predictions)}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 (macro): {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
