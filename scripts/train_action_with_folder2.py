#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_action_with_folder2.py — 将 2/ 中已知文件名标签的数据并入动作模型训练

注意：
    - 2/ 文件夹通过文件名解析 qpj/cpj/tj 标签。
    - Group03 不参与训练，只用于生成提交文件。
"""

import csv
import glob
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event_detection import detect_events
from src.features import extract_cycle_features
from src.preprocessing import preprocess
from src.utils import data_loader


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRS = [
    os.path.join(ROOT_DIR, "data"),
    os.path.join(ROOT_DIR, "2"),
]
GROUP03_DIR = os.path.join(ROOT_DIR, "Group03")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

ACTION_FEATURES = [
    "var_ch1",
    "rms_ch1",
    "iemg_ch1",
    "mav_ch1",
    "diff_rms",
    "ratio_rms",
    "rms_ratio_ch1",
    "rms_ratio_ch2",
    "rms_dom",
    "var_ratio",
    "rms_log_ratio",
    "var_ch2",
    "iemg_dom",
    "mav_dom",
    "rms_ch2",
]

ACTION_LABELS = {
    "qpj": "前平举",
    "cpj": "侧平举",
    "tj": "推肩",
}

SUBMIT_LABELS = {
    "前平举": 1,
    "侧平举": 2,
    "推肩": 3,
}


def parse_labeled_filename(filename):
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    if len(parts) < 4 or parts[0] != "emg":
        raise ValueError(f"无法从文件名解析标签: {filename}")
    subject_id = parts[1]
    action_code = parts[2]
    return subject_id, ACTION_LABELS[action_code]


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])


def extract_file_rows(path, subject_id=None, action_label=None):
    filename = os.path.basename(path)
    raw, fs = data_loader(path)
    filtered, _ = preprocess(
        raw,
        fs,
        apply_dc_removal=True,
        apply_amplitude_scaling=True,
        apply_notch=True,
        notch_freq=50,
        fc_high=20,
        fc_low=450,
    )
    events = detect_events(
        filtered,
        fs,
        smooth_window=0.3,
        min_duration=1.5,
        max_duration=8.0,
        target_k=5,
        activity_sigma=3.0,
    )

    rows = []
    for cycle_id, (start, end, _peak) in enumerate(events["cycles"], start=1):
        feat = extract_cycle_features(
            filtered[start:end + 1, 0],
            filtered[start:end + 1, 1],
            fs,
        )
        feat["filename"] = filename
        feat["subject_id"] = subject_id
        feat["cycle_id"] = cycle_id
        feat["action_label"] = action_label
        rows.append(feat)
    return rows


def load_training_data():
    rows = []
    file_summary = []
    for data_dir in DATA_DIRS:
        for path in sorted(glob.glob(os.path.join(data_dir, "emg_*.mat"))):
            filename = os.path.basename(path)
            subject_id, action_label = parse_labeled_filename(filename)
            file_rows = extract_file_rows(path, subject_id, action_label)
            rows.extend(file_rows)
            file_summary.append({
                "source": os.path.basename(data_dir),
                "filename": filename,
                "subject_id": subject_id,
                "action_label": action_label,
                "cycles": len(file_rows),
            })

    df = pd.DataFrame(rows)
    summary = pd.DataFrame(file_summary)
    return df, summary


def evaluate_leave_subject_out(df):
    X = df[ACTION_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    le = LabelEncoder()
    y = le.fit_transform(df["action_label"])
    subjects = df["subject_id"].to_numpy()

    y_true_all = []
    y_pred_all = []
    subject_rows = []
    for subject_id in sorted(np.unique(subjects)):
        train_mask = subjects != subject_id
        test_mask = subjects == subject_id
        if test_mask.sum() < 1:
            continue
        model = build_model()
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        y_true_all.extend(y[test_mask])
        y_pred_all.extend(pred)
        subject_rows.append({
            "subject_id": subject_id,
            "samples": int(test_mask.sum()),
            "accuracy": accuracy_score(y[test_mask], pred),
            "macro_f1": f1_score(y[test_mask], pred, average="macro"),
        })

    report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=le.classes_,
        digits=4,
    )
    return pd.DataFrame(subject_rows), report, le


def predict_group03(model, label_encoder, output_path):
    rows = []
    details = []
    for idx in range(1, 37):
        path = os.path.join(GROUP03_DIR, f"{idx}.mat")
        file_rows = extract_file_rows(path)
        feature_df = pd.DataFrame(file_rows)
        X = feature_df[ACTION_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        pred_encoded = model.predict(X)
        pred_labels = label_encoder.inverse_transform(pred_encoded)
        votes = Counter(pred_labels)
        action = votes.most_common(1)[0][0]
        submit_label = SUBMIT_LABELS[action]
        confidence = votes[action] / len(pred_labels)

        rows.append({"ID": idx, "Pred_Label": submit_label})
        details.append({
            "ID": idx,
            "Pred_Label": submit_label,
            "Action": action,
            "Cycles": len(pred_labels),
            "Confidence": round(confidence, 3),
            "Votes": dict(votes),
            "Cycle_Predictions": " | ".join(pred_labels),
        })

    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ID", "Pred_Label"])
        writer.writeheader()
        writer.writerows(rows)

    detail_path = os.path.join(OUTPUT_DIR, "group03_with2_prediction_details.csv")
    pd.DataFrame(details).to_csv(detail_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows), pd.DataFrame(details), detail_path


def main():
    print("=" * 70)
    print("  训练动作模型：data + 2")
    print("=" * 70)

    df, summary = load_training_data()
    print(f"训练周期样本: {len(df)}")
    print(f"训练文件数: {summary['filename'].nunique()}")
    print(f"类别分布: {df['action_label'].value_counts().to_dict()}")
    print(f"受试者: {sorted(summary['subject_id'].unique())}")

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "action_with2_training_files.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[1/3] Leave-subject-out 验证...")
    subject_eval, report, le = evaluate_leave_subject_out(df)
    print(subject_eval.to_string(index=False))
    print("\n分类报告:\n" + report)
    subject_eval.to_csv(
        os.path.join(OUTPUT_DIR, "action_with2_loso_by_subject.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    with open(os.path.join(OUTPUT_DIR, "action_with2_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("训练集: data + 2\n")
        fh.write(f"周期样本: {len(df)}\n")
        fh.write(f"类别分布: {df['action_label'].value_counts().to_dict()}\n\n")
        fh.write("Leave-subject-out:\n")
        fh.write(subject_eval.to_string(index=False))
        fh.write("\n\n分类报告:\n")
        fh.write(report)

    print("\n[2/3] 使用 data + 2 全量重训最终模型...")
    X = df[ACTION_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = le.transform(df["action_label"])
    final_model = build_model()
    final_model.fit(X, y)

    dump(final_model, os.path.join(MODEL_DIR, "action_model.joblib"))
    dump(le, os.path.join(MODEL_DIR, "label_encoder_action.joblib"))
    dump(ACTION_FEATURES, os.path.join(MODEL_DIR, "action_feature_names.joblib"))
    print("已保存 models/action_model.joblib")

    print("\n[3/3] 预测 Group03...")
    output_path = os.path.join(GROUP03_DIR, "Pred_Labels_Group03_Submit_with2.csv")
    submit_df, details, detail_path = predict_group03(final_model, le, output_path)
    print(f"提交文件: {output_path}")
    print(f"细节文件: {detail_path}")
    print(f"预测分布: {submit_df['Pred_Label'].value_counts().sort_index().to_dict()}")
    print(details[["ID", "Pred_Label", "Action", "Cycles", "Confidence", "Votes"]].to_string(index=False))


if __name__ == "__main__":
    main()
