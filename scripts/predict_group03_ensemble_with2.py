#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_group03_ensemble_with2.py — 使用 data + 2 训练多模型并预测 Group03
"""

import csv
import glob
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event_detection import detect_events
from src.features import extract_cycle_features
from src.preprocessing import preprocess
from src.utils import data_loader


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRS = [os.path.join(ROOT_DIR, "data"), os.path.join(ROOT_DIR, "2")]
GROUP03_DIR = os.path.join(ROOT_DIR, "Group03")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

FEATURES = [
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

CODE_TO_ACTION = {
    "qpj": "前平举",
    "cpj": "侧平举",
    "tj": "推肩",
}

ACTION_TO_SUBMIT = {
    "前平举": 1,
    "侧平举": 2,
    "推肩": 3,
}


def parse_labeled_filename(path):
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    return parts[1], parts[2]


def extract_rows(path, subject_id=None, action_code=None):
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
        feat["filename"] = os.path.basename(path)
        feat["subject_id"] = subject_id
        feat["action_code"] = action_code
        feat["cycle_id"] = cycle_id
        rows.append(feat)
    return rows


def load_training_df():
    rows = []
    for data_dir in DATA_DIRS:
        for path in sorted(glob.glob(os.path.join(data_dir, "emg_*.mat"))):
            subject_id, action_code = parse_labeled_filename(path)
            rows.extend(extract_rows(path, subject_id, action_code))
    return pd.DataFrame(rows)


def build_models():
    return {
        "LR": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clf", LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Ridge": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clf", RidgeClassifier(
                alpha=0.1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "SVC": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clf", SVC(
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
        "KNN": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", MinMaxScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=3,
                weights="distance",
                metric="manhattan",
            )),
        ]),
        "RF": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )),
        ]),
        "ET": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )),
        ]),
        "GBDT": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            )),
        ]),
        "MLP": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(50,),
                alpha=0.01,
                max_iter=1500,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def file_vote(model, x):
    cycle_pred = model.predict(x)
    vote_counts = Counter(cycle_pred)
    pred_code = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[pred_code] / len(cycle_pred)
    return pred_code, confidence, dict(vote_counts)


def main():
    print("=" * 70)
    print("  Group03 ensemble prediction trained on data + 2")
    print("=" * 70)

    train_df = load_training_df()
    x_train = train_df[FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y_train = train_df["action_code"].to_numpy()
    print(f"训练周期: {len(train_df)}")
    print(f"类别分布: {Counter(y_train)}")

    models = build_models()
    for name, model in models.items():
        model.fit(x_train, y_train)
        print(f"trained {name}")

    details = []
    submit_rows = []
    for idx in range(1, 37):
        path = os.path.join(GROUP03_DIR, f"{idx}.mat")
        rows = extract_rows(path)
        feature_df = pd.DataFrame(rows)
        x = feature_df[FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

        model_labels = {}
        model_conf = {}
        model_cycle_votes = {}
        for name, model in models.items():
            pred_code, conf, cycle_votes = file_vote(model, x)
            label = ACTION_TO_SUBMIT[CODE_TO_ACTION[pred_code]]
            model_labels[name] = label
            model_conf[f"{name}_conf"] = round(conf, 3)
            model_cycle_votes[f"{name}_cycle_votes"] = cycle_votes

        ensemble_counts = Counter(model_labels.values())
        ensemble_label = ensemble_counts.most_common(1)[0][0]
        ensemble_conf = ensemble_counts[ensemble_label] / len(models)

        submit_rows.append({"ID": idx, "Pred_Label": ensemble_label})
        details.append({
            "ID": idx,
            "Pred_Label": ensemble_label,
            "Cycles": len(rows),
            "Ensemble_Confidence": round(ensemble_conf, 3),
            "Ensemble_Votes": dict(sorted(ensemble_counts.items())),
            **model_labels,
            **model_conf,
            **model_cycle_votes,
        })

    submit_path = os.path.join(GROUP03_DIR, "Pred_Labels_Group03_EnsembleWith2.csv")
    with open(submit_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ID", "Pred_Label"])
        writer.writeheader()
        writer.writerows(submit_rows)

    details_df = pd.DataFrame(details)
    details_path = os.path.join(OUTPUT_DIR, "group03_ensemble_with2_details.csv")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    submit_df = pd.DataFrame(submit_rows)
    print(f"提交文件: {submit_path}")
    print(f"细节文件: {details_path}")
    print(f"分布: {submit_df['Pred_Label'].value_counts().sort_index().to_dict()}")
    print(details_df[[
        "ID", "Pred_Label", "Cycles", "Ensemble_Confidence", "Ensemble_Votes",
        "LR", "Ridge", "SVC", "KNN", "RF", "ET", "GBDT", "MLP",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
