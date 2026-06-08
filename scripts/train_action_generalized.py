#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_action_generalized.py — 泛化优先的动作类型三分类模型

目标：
    - 不读取 2/ 测试文件夹，避免测试集答案泄露
    - 使用按受试者留一验证（LOSO）选择更稳健的模型配置
    - 保存动作模型专用特征名，避免与质量模型 feature_names.joblib 冲突
"""

import glob
import os
import sys
import warnings

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
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# 该 15 维组合来自原动作模型特征列表；在当前数据上，配合
# RobustScaler + 强正则 LogisticRegression 的 LOSO 表现最稳。
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


def parse_filename(filename):
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    subject_id = parts[1]
    action_code = parts[2]
    return subject_id, ACTION_LABELS[action_code]


def load_training_rows():
    rows = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"加载训练文件: {len(files)}")

    for path in files:
        filename = os.path.basename(path)
        subject_id, action_label = parse_filename(filename)

        try:
            raw, fs = data_loader(path)
            filtered, _ = preprocess(raw, fs)
            events = detect_events(filtered, fs)
        except Exception as exc:
            print(f"  跳过 {filename}: {exc}")
            continue

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

    df = pd.DataFrame(rows)
    print(f"周期样本: {len(df)}")
    print(f"类别分布: {df['action_label'].value_counts().to_dict()}")
    print(f"受试者数: {df['subject_id'].nunique()}")
    return df


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


def evaluate_loso(X, y, subjects, label_encoder):
    y_true_all = []
    y_pred_all = []
    subject_rows = []

    for subject_id in sorted(np.unique(subjects)):
        train_mask = subjects != subject_id
        test_mask = subjects == subject_id

        model = build_model()
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])

        f1 = f1_score(y[test_mask], pred, average="macro")
        acc = accuracy_score(y[test_mask], pred)
        subject_rows.append({
            "test_subject": subject_id,
            "samples": int(test_mask.sum()),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
        })
        y_true_all.extend(y[test_mask])
        y_pred_all.extend(pred)

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=label_encoder.classes_,
        digits=4,
    )
    cm = confusion_matrix(y_true_all, y_pred_all)
    df_subject = pd.DataFrame(subject_rows)

    return {
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "macro_f1": f1_score(y_true_all, y_pred_all, average="macro"),
        "subject_mean_f1": df_subject["macro_f1"].mean(),
        "subject_std_f1": df_subject["macro_f1"].std(),
        "subject_min_f1": df_subject["macro_f1"].min(),
        "report": report,
        "confusion_matrix": cm,
        "subjects": df_subject,
    }


def main():
    print("=" * 70)
    print("  泛化优先动作分类模型训练")
    print("=" * 70)

    df = load_training_rows()
    missing = [name for name in ACTION_FEATURES if name not in df.columns]
    if missing:
        raise RuntimeError(f"缺失特征列: {missing}")

    X = df[ACTION_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    le = LabelEncoder()
    y = le.fit_transform(df["action_label"])
    subjects = df["subject_id"].to_numpy()

    print("\n[1/3] LOSO 受试者级验证...")
    loso = evaluate_loso(X, y, subjects, le)
    print(f"整体准确率: {loso['accuracy']:.4f}")
    print(f"整体 Macro F1: {loso['macro_f1']:.4f}")
    print(
        "受试者 Macro F1: "
        f"{loso['subject_mean_f1']:.4f} ± {loso['subject_std_f1']:.4f}, "
        f"min={loso['subject_min_f1']:.4f}"
    )
    print("\n分类报告:\n" + loso["report"])
    print("逐受试者结果:")
    print(loso["subjects"].to_string(index=False))

    print("\n[2/3] 使用全部训练数据拟合最终模型...")
    final_model = build_model()
    final_model.fit(X, y)

    print("\n[3/3] 保存模型与报告...")
    dump(final_model, os.path.join(MODEL_DIR, "action_model.joblib"))
    dump(le, os.path.join(MODEL_DIR, "label_encoder_action.joblib"))
    dump(ACTION_FEATURES, os.path.join(MODEL_DIR, "action_feature_names.joblib"))

    loso["subjects"].to_csv(
        os.path.join(OUTPUT_DIR, "action_generalized_loso_by_subject.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        os.path.join(OUTPUT_DIR, "action_generalized_report.txt"),
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write("泛化优先动作分类模型\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("模型: SimpleImputer(median) + RobustScaler + LogisticRegression(C=0.1, balanced)\n")
        fh.write(f"特征数: {len(ACTION_FEATURES)}\n")
        fh.write("特征:\n")
        for feat in ACTION_FEATURES:
            fh.write(f"- {feat}\n")
        fh.write("\nLOSO 结果:\n")
        fh.write(f"整体准确率: {loso['accuracy']:.4f}\n")
        fh.write(f"整体 Macro F1: {loso['macro_f1']:.4f}\n")
        fh.write(
            "受试者 Macro F1: "
            f"{loso['subject_mean_f1']:.4f} ± {loso['subject_std_f1']:.4f}, "
            f"min={loso['subject_min_f1']:.4f}\n\n"
        )
        fh.write("分类报告:\n")
        fh.write(loso["report"])
        fh.write("\n逐受试者结果:\n")
        fh.write(loso["subjects"].to_string(index=False))
        fh.write("\n")

    print("已保存:")
    print("  models/action_model.joblib")
    print("  models/label_encoder_action.joblib")
    print("  models/action_feature_names.joblib")
    print("  outputs/action_generalized_report.txt")
    print("  outputs/action_generalized_loso_by_subject.csv")


if __name__ == "__main__":
    main()
