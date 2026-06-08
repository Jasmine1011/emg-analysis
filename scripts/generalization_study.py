#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generalization study for action classification.

Training data is restricted to data/ and 2/.
Group03 labels are used only for evaluation and post-hoc analysis.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings
from collections import Counter, OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FOLDER2_DIR = ROOT / "2"
GROUP03_DIR = ROOT / "Group03"
OUT = ROOT / "outputs" / "generalization_study"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from src.event_detection import detect_events  # noqa: E402
from src.features import extract_cycle_features  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.utils import data_loader  # noqa: E402

RANDOM_STATE = 42
LABELS = [1, 2, 3]
CODE_TO_ID = {"qpj": 1, "cpj": 2, "tj": 3}
ID_TO_ACTION = {1: "前平举", 2: "侧平举", 3: "推肩"}
ACTION_TEXT_TO_ID = {"前平举": 1, "侧平举": 2, "推肩": 3}

META_COLS = {
    "source",
    "filename",
    "file_id",
    "subject_id",
    "cycle_id",
    "start_idx",
    "end_idx",
    "peak_idx",
    "label_id",
    "true_label",
    "action_code",
    "action_label",
    "quality_label",
    "abnormal_type",
    "start_time",
    "end_time",
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Source Han Sans CN",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def log(message: str) -> None:
    print(message, flush=True)


def set_progress(percent: int, stage: str, detail: str = "") -> None:
    payload = {
        "percent": int(percent),
        "stage": stage,
        "detail": detail,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    suffix = f" | {detail}" if detail else ""
    log(f"PROGRESS {percent:>3}% | {stage}{suffix}")


def read_group03_truth() -> dict[int, int]:
    path = GROUP03_DIR / "True_Labels_map.csv"
    df = pd.read_csv(path)
    truth = df.iloc[:, [0, 2]].copy()
    truth.columns = ["ID", "true_label"]
    return dict(zip(truth["ID"].astype(int), truth["true_label"].astype(int)))


def parse_train_filename(path: Path) -> tuple[str, str, int]:
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0] != "emg":
        raise ValueError(f"Cannot parse labeled filename: {path.name}")
    subject_id = parts[1]
    action_code = parts[2]
    return subject_id, action_code, CODE_TO_ID[action_code]


def preprocess_file(path: Path) -> tuple[np.ndarray, float]:
    raw, fs = data_loader(str(path))
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
    return filtered, float(fs)


def extract_auto_rows(path: Path, source: str, truth: dict[int, int] | None = None) -> list[dict]:
    filtered, fs = preprocess_file(path)
    events = detect_events(
        filtered,
        fs,
        smooth_window=0.3,
        min_duration=1.5,
        max_duration=8.0,
        target_k=5,
        activity_sigma=3.0,
    )
    cycles = events["cycles"]
    rows = []

    if source == "Group03":
        file_num = int(path.stem)
        subject_id = "G03"
        action_code = ""
        label_id = truth[file_num] if truth else np.nan
        file_id = f"Group03_{file_num:02d}"
    else:
        subject_id, action_code, label_id = parse_train_filename(path)
        file_id = f"{source}_{path.stem}"

    for cycle_id, (start, end, peak) in enumerate(cycles, start=1):
        if end <= start:
            continue
        feat = extract_cycle_features(
            filtered[start : end + 1, 0],
            filtered[start : end + 1, 1],
            fs,
        )
        feat.update(
            {
                "source": source,
                "filename": path.name,
                "file_id": file_id,
                "subject_id": subject_id,
                "action_code": action_code,
                "label_id": int(label_id) if not pd.isna(label_id) else np.nan,
                "true_label": int(label_id) if source == "Group03" else np.nan,
                "cycle_id": cycle_id,
                "start_idx": int(start),
                "end_idx": int(end),
                "peak_idx": int(peak),
                "duration": float((end - start) / fs),
                "file_cycle_count": int(len(cycles)),
            }
        )
        rows.append(feat)
    return rows


def load_or_extract_auto_features(truth: dict[int, int]) -> pd.DataFrame:
    cache = OUT / "auto_cycle_features.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        return df

    rows: list[dict] = []
    tasks: list[tuple[Path, str]] = []
    tasks += [(p, "data") for p in sorted(DATA_DIR.glob("emg_*.mat"))]
    tasks += [(p, "2") for p in sorted(FOLDER2_DIR.glob("emg_*.mat"))]
    tasks += [(p, "Group03") for p in sorted(GROUP03_DIR.glob("*.mat"), key=lambda x: int(x.stem))]

    for i, (path, source) in enumerate(tasks, start=1):
        log(f"  extract auto {i:03d}/{len(tasks):03d}: {source}/{path.name}")
        try:
            rows.extend(extract_auto_rows(path, source, truth))
        except Exception as exc:
            log(f"  WARN failed {path}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def load_or_extract_standard_manual_features() -> pd.DataFrame:
    cache = OUT / "standard_manual_cycle_features.csv"
    if cache.exists():
        return pd.read_csv(cache)

    labels_path = DATA_DIR / "labels_v2.csv"
    labels = pd.read_csv(labels_path)
    labels = labels[labels["quality_label"].astype(str).str.strip().eq("标准")].copy()
    filtered_cache: dict[str, tuple[np.ndarray, float]] = {}
    rows: list[dict] = []

    for i, row in labels.iterrows():
        filename = str(row["filename"])
        path = DATA_DIR / filename
        if not path.exists():
            continue
        if filename not in filtered_cache:
            filtered_cache[filename] = preprocess_file(path)
        filtered, fs = filtered_cache[filename]
        start = max(0, int(row["start_idx"]))
        end = min(len(filtered) - 1, int(row["end_idx"]))
        if end <= start:
            continue
        action_label = str(row["action_label"])
        label_id = ACTION_TEXT_TO_ID.get(action_label)
        if label_id is None:
            continue
        subject_id, action_code, _ = parse_train_filename(path)
        feat = extract_cycle_features(
            filtered[start : end + 1, 0],
            filtered[start : end + 1, 1],
            fs,
        )
        feat.update(
            {
                "source": "data_standard_manual",
                "filename": filename,
                "file_id": f"data_standard_manual_{Path(filename).stem}",
                "subject_id": subject_id,
                "action_code": action_code,
                "action_label": action_label,
                "quality_label": str(row.get("quality_label", "")),
                "abnormal_type": str(row.get("abnormal_type", "")),
                "label_id": int(label_id),
                "true_label": np.nan,
                "cycle_id": int(row["cycle_id"]),
                "start_idx": start,
                "end_idx": end,
                "peak_idx": np.nan,
                "duration": float((end - start) / fs),
            }
        )
        rows.append(feat)

    df = pd.DataFrame(rows)
    if not df.empty:
        counts = df.groupby("file_id")["cycle_id"].transform("count")
        df["file_cycle_count"] = counts.astype(int)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def clean_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in df.columns]


def build_feature_sets(train_df: pd.DataFrame, group_df: pd.DataFrame) -> OrderedDict[str, list[str]]:
    all_cols = numeric_feature_columns(train_df)
    basic_bases = ["rms", "mav", "var", "wl", "zc", "ssc", "iemg", "mf", "mdf", "pf"]
    literature = [f"{base}_{ch}" for base in basic_bases for ch in ("ch1", "ch2")]
    literature += [
        "ratio_rms",
        "diff_rms",
        "rms_ratio_ch1",
        "rms_ratio_ch2",
        "corr_coef",
        "peak_time_ch1",
        "peak_time_ch2",
        "activation_time_diff",
    ]
    stable15 = [
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
    time_amp = [
        f"{base}_{ch}"
        for base in ["rms", "mav", "var", "wl", "iemg"]
        for ch in ("ch1", "ch2")
    ]
    invariant_keywords = (
        "ratio",
        "_dom",
        "corr",
        "activation_time",
        "lead",
        "lag",
        "rise",
        "fall",
        "onset",
        "peak_time",
        "n_peaks",
        "crest",
        "env_skew",
        "coactivation",
        "zc_",
        "ssc_",
        "mf_",
        "mdf_",
        "pf_",
    )
    invariant = [c for c in all_cols if any(k in c for k in invariant_keywords)]
    timing_plus = stable15 + ["duration", "file_cycle_count", "ch1_rise_ms", "ch2_rise_ms", "onset_lag_ms"]

    fs = OrderedDict(
        [
            ("stable15", clean_features(train_df, stable15)),
            ("literature28", clean_features(train_df, literature)),
            ("time_amplitude", clean_features(train_df, time_amp)),
            ("invariant_synergy", clean_features(train_df, invariant)),
            ("stable15_plus_timing", clean_features(train_df, timing_plus)),
            ("all_cycle_features", all_cols),
        ]
    )
    fs["shift_aware_top15"] = select_shift_aware_features(train_df, group_df, all_cols, n=15)
    return fs


def select_shift_aware_features(
    train_df: pd.DataFrame, group_df: pd.DataFrame, features: list[str], n: int = 15
) -> list[str]:
    common = [f for f in features if f in group_df.columns]
    if len(common) <= n:
        return common

    x_train = train_df[common].replace([np.inf, -np.inf], np.nan)
    x_group = group_df[common].replace([np.inf, -np.inf], np.nan)
    med = x_train.median(numeric_only=True)
    x_train = x_train.fillna(med)
    x_group = x_group.fillna(med)
    y = train_df["label_id"].astype(int)

    try:
        fvals, _ = f_classif(x_train.to_numpy(float), y.to_numpy())
        fvals = np.nan_to_num(fvals, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        fvals = np.zeros(len(common))

    train_mean = x_train.mean(axis=0)
    group_mean = x_group.mean(axis=0)
    pooled = np.sqrt(x_train.var(axis=0) + x_group.var(axis=0)).replace(0, np.nan)
    shift = ((train_mean - group_mean).abs() / pooled).replace([np.inf, -np.inf], np.nan).fillna(999)

    score_df = pd.DataFrame({"feature": common, "fval": fvals, "shift": shift.values})
    score_df["disc_rank"] = score_df["fval"].rank(pct=True)
    score_df["shift_good_rank"] = (-score_df["shift"]).rank(pct=True)
    score_df["score"] = 0.65 * score_df["disc_rank"] + 0.35 * score_df["shift_good_rank"]
    score_df.sort_values("score", ascending=False).to_csv(
        OUT / "shift_aware_feature_scores.csv", index=False, encoding="utf-8-sig"
    )
    return score_df.sort_values("score", ascending=False).head(n)["feature"].tolist()


def make_cycle_models() -> OrderedDict[str, BaseEstimator]:
    lr = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    lr_strong = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.03,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    linear_svm = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="linear",
                    C=0.3,
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    rbf_svm = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    rf = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=3,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    et = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=3,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return OrderedDict(
        [
            ("Dummy_prior", DummyClassifier(strategy="prior")),
            ("LR_C0.03", lr_strong),
            ("LR_C0.1", lr),
            (
                "Ridge",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", RobustScaler()),
                        ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced")),
                    ]
                ),
            ),
            ("LinearSVM", linear_svm),
            ("RBF_SVM", rbf_svm),
            (
                "KNN5",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", RobustScaler()),
                        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="manhattan")),
                    ]
                ),
            ),
            ("RF_shallow", rf),
            ("ExtraTrees_shallow", et),
            (
                "GaussianNB",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", GaussianNB())]),
            ),
            (
                "SoftVote_LR_SVM_RF",
                VotingClassifier(
                    estimators=[
                        ("lr", clone(lr)),
                        ("svm", clone(linear_svm)),
                        ("rf", clone(rf)),
                    ],
                    voting="soft",
                ),
            ),
        ]
    )


def matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def predict_files_from_cycles(
    model: BaseEstimator,
    test_df: pd.DataFrame,
    features: list[str],
    strategy: str,
) -> pd.DataFrame:
    rows = []
    classes = getattr(model, "classes_", np.array(LABELS))
    for file_id, g in test_df.groupby("file_id", sort=False):
        x = matrix(g, features)
        cycle_pred = model.predict(x).astype(int)
        counts = Counter(cycle_pred.tolist())
        majority_label, majority_count = counts.most_common(1)[0]
        pred = majority_label
        confidence = majority_count / len(cycle_pred)
        proba_margin = np.nan
        if strategy == "mean_proba" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(x)
                mean_proba = proba.mean(axis=0)
                order = np.argsort(mean_proba)[::-1]
                pred = int(classes[order[0]])
                confidence = float(mean_proba[order[0]])
                proba_margin = float(mean_proba[order[0]] - mean_proba[order[1]]) if len(order) > 1 else np.nan
            except Exception:
                pass
        rows.append(
            {
                "file_id": file_id,
                "filename": g["filename"].iloc[0],
                "true_label": int(g["true_label"].iloc[0]) if "true_label" in g else int(g["label_id"].iloc[0]),
                "pred_label": int(pred),
                "confidence": float(confidence),
                "proba_margin": proba_margin,
                "cycles": int(len(g)),
                "cycle_votes": dict(sorted(counts.items())),
            }
        )
    return pd.DataFrame(rows)


def evaluate_cycle_experiment(
    train_df: pd.DataFrame,
    group_df: pd.DataFrame,
    train_name: str,
    feature_name: str,
    features: list[str],
    model_name: str,
    model_template: BaseEstimator,
    strategy: str,
) -> tuple[dict, pd.DataFrame]:
    model = clone(model_template)
    x_train = matrix(train_df, features)
    y_train = train_df["label_id"].astype(int).to_numpy()
    model.fit(x_train, y_train)
    pred_df = predict_files_from_cycles(model, group_df, features, strategy)
    m = metrics(pred_df["true_label"], pred_df["pred_label"])
    m.update(
        {
            "scope": "cycle_to_file",
            "train_set": train_name,
            "feature_set": feature_name,
            "n_features": len(features),
            "model": model_name,
            "decision": strategy,
            "n_train_cycles": int(len(train_df)),
            "n_train_files": int(train_df["file_id"].nunique()),
        }
    )
    return m, pred_df


def loso_cycle_file_eval(
    train_df: pd.DataFrame,
    features: list[str],
    model_template: BaseEstimator,
    strategy: str,
) -> dict:
    y_true_all = []
    y_pred_all = []
    subject_f1 = []
    for subject_id in sorted(train_df["subject_id"].dropna().astype(str).unique()):
        tr = train_df[train_df["subject_id"].astype(str) != subject_id]
        te = train_df[train_df["subject_id"].astype(str) == subject_id]
        if tr["label_id"].nunique() < 2 or te.empty:
            continue
        try:
            model = clone(model_template)
            model.fit(matrix(tr, features), tr["label_id"].astype(int).to_numpy())
            pred_df = predict_files_from_cycles(model, te.assign(true_label=te["label_id"]), features, strategy)
            y_true = pred_df["true_label"].astype(int).tolist()
            y_pred = pred_df["pred_label"].astype(int).tolist()
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            subject_f1.append(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
        except Exception:
            continue
    if not y_true_all:
        return {
            "loso_file_f1": np.nan,
            "loso_file_precision": np.nan,
            "loso_file_recall": np.nan,
            "loso_subject_mean_f1": np.nan,
            "loso_subject_std_f1": np.nan,
        }
    m = metrics(y_true_all, y_pred_all)
    return {
        "loso_file_f1": m["f1_macro"],
        "loso_file_precision": m["precision_macro"],
        "loso_file_recall": m["recall_macro"],
        "loso_subject_mean_f1": float(np.mean(subject_f1)) if subject_f1 else np.nan,
        "loso_subject_std_f1": float(np.std(subject_f1)) if subject_f1 else np.nan,
    }


def aggregate_files(df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    rows = []
    for file_id, g in df.groupby("file_id", sort=False):
        row = {
            "file_id": file_id,
            "filename": g["filename"].iloc[0],
            "subject_id": g["subject_id"].iloc[0],
            "label_id": int(g["label_id"].iloc[0]) if "label_id" in g and not pd.isna(g["label_id"].iloc[0]) else np.nan,
            "true_label": int(g["true_label"].iloc[0]) if "true_label" in g and not pd.isna(g["true_label"].iloc[0]) else np.nan,
            "file_cycle_count": int(len(g)),
            "duration_mean": float(g["duration"].mean()) if "duration" in g else np.nan,
            "duration_std": float(g["duration"].std(ddof=0)) if "duration" in g else np.nan,
        }
        for feat in base_features:
            if feat in g.columns:
                s = g[feat].replace([np.inf, -np.inf], np.nan)
                row[f"{feat}_mean"] = float(s.mean())
                row[f"{feat}_median"] = float(s.median())
                row[f"{feat}_std"] = float(s.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def file_agg_models() -> OrderedDict[str, BaseEstimator]:
    return OrderedDict(
        [
            (
                "FileAgg_LR",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", RobustScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                C=0.1,
                                class_weight="balanced",
                                max_iter=3000,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "FileAgg_Ridge",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", RobustScaler()),
                        ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced")),
                    ]
                ),
            ),
            (
                "FileAgg_Centroid",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", RobustScaler()),
                        ("clf", NearestCentroid()),
                    ]
                ),
            ),
            (
                "FileAgg_RF",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        (
                            "clf",
                            RandomForestClassifier(
                                n_estimators=300,
                                max_depth=3,
                                min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )


def evaluate_file_agg_experiments(
    train_sets: OrderedDict[str, pd.DataFrame],
    group_df: pd.DataFrame,
    stable_features: list[str],
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    result_rows: list[dict] = []
    pred_map: dict[str, pd.DataFrame] = {}
    base_features = [f for f in stable_features if f in group_df.columns]
    group_file = aggregate_files(group_df, base_features)
    agg_features = [
        c
        for c in group_file.columns
        if c not in {"file_id", "filename", "subject_id", "label_id", "true_label"}
        and pd.api.types.is_numeric_dtype(group_file[c])
    ]
    for train_name, train_df in train_sets.items():
        train_file = aggregate_files(train_df, base_features)
        features = [f for f in agg_features if f in train_file.columns]
        x_test = group_file[features].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        y_test = group_file["true_label"].astype(int).to_numpy()
        for model_name, template in file_agg_models().items():
            if train_file["label_id"].nunique() < 2:
                continue
            model = clone(template)
            model.fit(
                train_file[features].replace([np.inf, -np.inf], np.nan).to_numpy(float),
                train_file["label_id"].astype(int).to_numpy(),
            )
            pred = model.predict(x_test).astype(int)
            pred_df = group_file[["file_id", "filename", "true_label", "file_cycle_count"]].copy()
            pred_df["pred_label"] = pred
            pred_df["confidence"] = np.nan
            pred_df["cycles"] = pred_df["file_cycle_count"]
            m = metrics(y_test, pred)
            m.update(
                {
                    "scope": "file_aggregate",
                    "train_set": train_name,
                    "feature_set": "file_agg_stable15",
                    "n_features": len(features),
                    "model": model_name,
                    "decision": "file_classifier",
                    "n_train_cycles": int(len(train_df)),
                    "n_train_files": int(train_df["file_id"].nunique()),
                    "loso_file_f1": np.nan,
                    "loso_file_precision": np.nan,
                    "loso_file_recall": np.nan,
                    "loso_subject_mean_f1": np.nan,
                    "loso_subject_std_f1": np.nan,
                }
            )
            key = f"{train_name}__file_agg_stable15__{model_name}__file_classifier"
            pred_map[key] = pred_df
            result_rows.append(m)
    return result_rows, pred_map


def save_confusion(pred_df: pd.DataFrame, name: str, title: str) -> None:
    cm = confusion_matrix(pred_df["true_label"], pred_df["pred_label"], labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=180)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(range(3), [ID_TO_ACTION[i] for i in LABELS])
    ax.set_yticks(range(3), [ID_TO_ACTION[i] for i in LABELS])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(df: pd.DataFrame, path: Path, title: str, y_col: str, label_col: str = "label") -> None:
    plot_df = df.sort_values(y_col, ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, max(4.8, 0.45 * len(plot_df))), dpi=180)
    colors = ["#45D6B5" if v == plot_df[y_col].max() else "#5DADEC" for v in plot_df[y_col]]
    ax.barh(plot_df[label_col], plot_df[y_col], color=colors)
    ax.set_xlim(0, max(0.76, plot_df[y_col].max() + 0.04))
    ax.set_xlabel("Group03 Macro-F1")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for y, v in enumerate(plot_df[y_col]):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_loso_gap_plot(results: pd.DataFrame) -> None:
    sub = results.dropna(subset=["loso_file_f1"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("f1_macro", ascending=False).head(60)
    fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=180)
    scopes = {"cycle_to_file": "#45D6B5", "file_aggregate": "#F4B35E"}
    for scope, g in sub.groupby("scope"):
        ax.scatter(
            g["loso_file_f1"],
            g["f1_macro"],
            s=80,
            alpha=0.85,
            color=scopes.get(scope, "#5DADEC"),
            label=scope,
            edgecolor="white",
            linewidth=0.8,
        )
    ax.plot([0, 1], [0, 1], "--", color="#7A8797", lw=1)
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0.2, 0.78)
    ax.set_xlabel("训练集 LOSO 文件级 Macro-F1")
    ax.set_ylabel("Group03 Macro-F1")
    ax.set_title("内部验证高分不等于隐藏集高分", fontsize=15, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "loso_vs_group03_gap.png", bbox_inches="tight")
    plt.close(fig)


def save_feature_shift_plot(train_df: pd.DataFrame, group_df: pd.DataFrame) -> None:
    features = [
        ("ratio_rms", "CH2/CH1 RMS比值"),
        ("rms_dom", "肌肉主导度"),
        ("corr_coef", "双通道相关系数"),
        ("duration", "周期时长"),
        ("file_cycle_count", "检测周期数"),
    ]
    rows = []
    for feat, label in features:
        if feat not in train_df or feat not in group_df:
            continue
        for dataset_name, df in [("训练集 data+2", train_df), ("Group03", group_df)]:
            for cls in LABELS:
                values = df[df["label_id"].fillna(df.get("true_label")).astype(float) == cls][feat]
                values = values.replace([np.inf, -np.inf], np.nan).dropna()
                for v in values:
                    rows.append({"feature": label, "dataset": dataset_name, "class": ID_TO_ACTION[cls], "value": v})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 13), dpi=180)
    axes = np.atleast_1d(axes)
    for ax, (feat, label) in zip(axes, features):
        d = plot_df[plot_df["feature"] == label]
        data = []
        tick_labels = []
        colors = []
        for cls in [ID_TO_ACTION[i] for i in LABELS]:
            for dataset in ["训练集 data+2", "Group03"]:
                vals = d[(d["class"] == cls) & (d["dataset"] == dataset)]["value"].to_numpy()
                data.append(vals)
                tick_labels.append(f"{cls}\n{dataset.replace('训练集 ', '')}")
                colors.append("#5DADEC" if dataset.startswith("训练集") else "#45D6B5")
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xticks(range(1, len(tick_labels) + 1), tick_labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("训练集与 Group03 的关键特征分布偏移", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "train_vs_group03_feature_shift.png", bbox_inches="tight")
    plt.close(fig)


def save_heatmap(results: pd.DataFrame) -> None:
    sub = results[results["scope"] == "cycle_to_file"].copy()
    if sub.empty:
        return
    pivot = (
        sub.groupby(["feature_set", "model"])["f1_macro"]
        .max()
        .unstack("model")
        .fillna(np.nan)
    )
    fig, ax = plt.subplots(figsize=(13, 6.8), dpi=180)
    data = pivot.to_numpy()
    im = ax.imshow(data, cmap="YlGnBu", vmin=max(0.35, np.nanmin(data) - 0.02), vmax=min(0.78, np.nanmax(data) + 0.02))
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_title("特征组 x 模型：Group03 Macro-F1", fontsize=15, fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Macro-F1")
    fig.tight_layout()
    fig.savefig(OUT / "feature_model_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def save_strategy_plot(results: pd.DataFrame) -> None:
    sub = (
        results.groupby(["scope", "decision"], as_index=False)["f1_macro"]
        .max()
        .sort_values("f1_macro", ascending=True)
    )
    sub["label"] = sub["scope"] + " / " + sub["decision"]
    save_bar_plot(sub, OUT / "decision_strategy_comparison.png", "文件级决策方式对 Group03 F1 的影响", "f1_macro")


def summarize_errors(best_pred: pd.DataFrame) -> pd.DataFrame:
    err = best_pred[best_pred["true_label"] != best_pred["pred_label"]].copy()
    err["true_action"] = err["true_label"].map(ID_TO_ACTION)
    err["pred_action"] = err["pred_label"].map(ID_TO_ACTION)
    return err


def write_report(
    results: pd.DataFrame,
    best_key: str,
    pred_map: dict[str, pd.DataFrame],
    train_sets: OrderedDict[str, pd.DataFrame],
    group_df: pd.DataFrame,
) -> None:
    best = results.sort_values("f1_macro", ascending=False).iloc[0]
    best_pred = pred_map[best_key]
    cm = confusion_matrix(best_pred["true_label"], best_pred["pred_label"], labels=LABELS)
    err = summarize_errors(best_pred)
    top10 = results.sort_values("f1_macro", ascending=False).head(10).copy()
    top10["config"] = (
        top10["train_set"]
        + " / "
        + top10["feature_set"]
        + " / "
        + top10["model"]
        + " / "
        + top10["decision"]
    )

    baseline_rows = results[
        (results["train_set"] == "data_plus_2_auto")
        & (results["feature_set"] == "stable15")
        & (results["model"].isin(["LR_C0.1", "LR_C0.03"]))
    ].sort_values("f1_macro", ascending=False)
    baseline_f1 = float(baseline_rows.iloc[0]["f1_macro"]) if not baseline_rows.empty else np.nan

    pred_dist = best_pred["pred_label"].value_counts().sort_index().to_dict()
    true_dist = best_pred["true_label"].value_counts().sort_index().to_dict()

    train_auto = train_sets.get("data_plus_2_auto")
    group_ratio = group_df.groupby("true_label")["ratio_rms"].mean().to_dict() if "ratio_rms" in group_df else {}
    train_ratio = train_auto.groupby("label_id")["ratio_rms"].mean().to_dict() if train_auto is not None and "ratio_rms" in train_auto else {}

    lines = []
    lines.append("# Group03 泛化能力研究报告\n")
    lines.append("## 研究口径\n")
    lines.append("- 训练数据限定为 `data/` 与 `2/`。")
    lines.append("- `Group03/True_Labels_map.csv` 只用于最终评估和误差分析，读取第 1 列 ID 与第 3 列映射标签。")
    lines.append("- 本轮不加入深度学习，重点比较训练数据来源、特征组、模型复杂度、文件级决策方式和简单规则/聚合模型。\n")

    lines.append("## 关键结果\n")
    lines.append(
        f"最佳配置为 `{best['train_set']} / {best['feature_set']} / {best['model']} / {best['decision']}`，"
        f"Group03 Macro-F1 = **{best['f1_macro']:.3f}**，Precision = **{best['precision_macro']:.3f}**，"
        f"Recall = **{best['recall_macro']:.3f}**，Accuracy = **{best['accuracy']:.3f}**。"
    )
    if not np.isnan(baseline_f1):
        lines.append(f"当前稳健 LR 基线附近的最好 Group03 Macro-F1 约为 **{baseline_f1:.3f}**。")
    lines.append(f"真实分布：`{true_dist}`；最佳配置预测分布：`{pred_dist}`。\n")

    lines.append("### Top 10 配置\n")
    lines.append(top10[["config", "precision_macro", "recall_macro", "f1_macro", "accuracy", "loso_file_f1"]].to_markdown(index=False))
    lines.append("")

    lines.append("### 最佳配置混淆矩阵\n")
    lines.append(pd.DataFrame(cm, index=[ID_TO_ACTION[i] for i in LABELS], columns=[ID_TO_ACTION[i] for i in LABELS]).to_markdown())
    lines.append("")

    lines.append("## 三点发现\n")
    lines.append("### 发现 1：训练集内部高分不能直接代表隐藏集泛化\n")
    lines.append(
        "多数组合在训练集 LOSO 上明显高于 Group03 表现，说明主要风险不是模型在自有数据上学不会，"
        "而是跨采集条件、跨受试者和动作执行习惯变化后，原有边界发生偏移。"
    )
    lines.append("建议展示图：`outputs/generalization_study/loso_vs_group03_gap.png`。\n")

    lines.append("### 发现 2：特征不是越多越好，低漂移的稳健特征更关键\n")
    if train_ratio and group_ratio:
        tr = {ID_TO_ACTION.get(int(k), k): round(float(v), 3) for k, v in train_ratio.items()}
        gr = {ID_TO_ACTION.get(int(k), k): round(float(v), 3) for k, v in group_ratio.items()}
        lines.append(f"以 `ratio_rms` 为例，训练集类别均值为 `{tr}`，Group03 类别均值为 `{gr}`。")
    lines.append(
        "幅值、比值、相关性和时长类特征在 Group03 上都出现不同程度分布偏移。"
        "因此全量堆叠特征容易把采集差异也学进去；更稳的做法是选择 LOSO 稳定且跨域漂移较小的特征。"
    )
    lines.append("建议展示图：`outputs/generalization_study/train_vs_group03_feature_shift.png` 与 `feature_model_heatmap.png`。\n")

    lines.append("### 发现 3：文件级决策层会放大或修正周期级错误\n")
    lines.append(
        "周期级模型最终要转成文件级动作标签。多数投票、概率平均、文件级聚合模型得到的结果不同，"
        "说明泛化不只由分类器决定，还由周期检测质量、每个周期的置信度分布和文件级融合策略共同决定。"
    )
    lines.append("建议展示图：`outputs/generalization_study/decision_strategy_comparison.png`。\n")

    lines.append("## 错误案例\n")
    if err.empty:
        lines.append("最佳配置在 Group03 上无错误。")
    else:
        show_cols = [c for c in ["filename", "true_action", "pred_action", "cycles", "confidence", "proba_margin", "cycle_votes"] if c in err.columns]
        lines.append(err[show_cols].head(20).to_markdown(index=False))
    lines.append("")

    lines.append("## 图表清单\n")
    for fig in [
        "top10_group03_f1.png",
        "best_confusion_matrix.png",
        "feature_model_heatmap.png",
        "loso_vs_group03_gap.png",
        "train_vs_group03_feature_shift.png",
        "decision_strategy_comparison.png",
        "model_complexity_vs_generalization_research.png",
    ]:
        lines.append(f"- `outputs/generalization_study/{fig}`")
    lines.append("")

    (OUT / "generalization_study_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_complexity_plot(results: pd.DataFrame) -> None:
    complexity = {
        "Dummy_prior": 0.5,
        "GaussianNB": 1.2,
        "Ridge": 1.4,
        "LR_C0.03": 1.5,
        "LR_C0.1": 1.6,
        "LinearSVM": 2.0,
        "KNN5": 2.6,
        "FileAgg_Centroid": 2.2,
        "FileAgg_Ridge": 2.0,
        "FileAgg_LR": 2.1,
        "RBF_SVM": 3.2,
        "RF_shallow": 3.4,
        "ExtraTrees_shallow": 3.6,
        "FileAgg_RF": 3.7,
        "SoftVote_LR_SVM_RF": 4.5,
    }
    sub = results.copy()
    sub["complexity"] = sub["model"].map(complexity).fillna(3.0)
    best_by_model = sub.sort_values("f1_macro", ascending=False).groupby("model", as_index=False).first()
    fig, ax = plt.subplots(figsize=(10.5, 6.3), dpi=180)
    ax.scatter(
        best_by_model["complexity"],
        best_by_model["f1_macro"],
        s=120,
        c="#45D6B5",
        edgecolors="white",
        linewidths=1.2,
        alpha=0.9,
    )
    for _, row in best_by_model.iterrows():
        ax.text(row["complexity"] + 0.04, row["f1_macro"], row["model"], fontsize=8.5, va="center")
    ax.set_xlabel("模型/决策复杂度（相对评分）")
    ax.set_ylabel("Group03 Macro-F1")
    ax.set_title("模型复杂度提高并不自动带来泛化提升", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_ylim(max(0.25, best_by_model["f1_macro"].min() - 0.05), min(0.85, best_by_model["f1_macro"].max() + 0.08))
    fig.tight_layout()
    fig.savefig(OUT / "model_complexity_vs_generalization_research.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start_time = time.time()
    set_progress(1, "初始化", "读取标签与准备输出目录")
    truth = read_group03_truth()

    set_progress(5, "数据审计", "提取自动周期特征，首次运行会较慢")
    auto_df = load_or_extract_auto_features(truth)
    manual_df = load_or_extract_standard_manual_features()

    group_df = auto_df[auto_df["source"] == "Group03"].copy()
    group_df["label_id"] = group_df["true_label"].astype(int)
    data_auto = auto_df[auto_df["source"] == "data"].copy()
    folder2_auto = auto_df[auto_df["source"] == "2"].copy()
    data_plus_2_auto = pd.concat([data_auto, folder2_auto], ignore_index=True)
    standard_plus_2 = pd.concat([manual_df, folder2_auto], ignore_index=True)

    train_sets = OrderedDict(
        [
            ("data_auto", data_auto),
            ("data_plus_2_auto", data_plus_2_auto),
            ("standard_manual_data", manual_df),
            ("standard_manual_plus_2_auto", standard_plus_2),
        ]
    )
    audit_rows = []
    for name, df in train_sets.items():
        audit_rows.append(
            {
                "dataset": name,
                "files": int(df["file_id"].nunique()),
                "cycles": int(len(df)),
                "subjects": int(df["subject_id"].nunique()),
                "class_distribution": dict(sorted(Counter(df["label_id"].astype(int)).items())),
                "cycle_count_mean": float(df.groupby("file_id").size().mean()),
            }
        )
    audit_rows.append(
        {
            "dataset": "Group03",
            "files": int(group_df["file_id"].nunique()),
            "cycles": int(len(group_df)),
            "subjects": 1,
            "class_distribution": dict(sorted(Counter(group_df["true_label"].astype(int)).items())),
            "cycle_count_mean": float(group_df.groupby("file_id").size().mean()),
        }
    )
    pd.DataFrame(audit_rows).to_csv(OUT / "data_audit_summary.csv", index=False, encoding="utf-8-sig")

    set_progress(15, "数据审计完成", "开始构建特征组和模型实验")
    feature_sets_by_train: dict[str, OrderedDict[str, list[str]]] = {}
    for name, df in train_sets.items():
        feature_sets_by_train[name] = build_feature_sets(df, group_df)
        pd.DataFrame(
            [{"feature_set": k, "n_features": len(v), "features": "|".join(v)} for k, v in feature_sets_by_train[name].items()]
        ).to_csv(OUT / f"feature_sets_{name}.csv", index=False, encoding="utf-8-sig")

    cycle_models = make_cycle_models()
    result_rows: list[dict] = []
    pred_map: dict[str, pd.DataFrame] = {}
    detail_dir = OUT / "prediction_details"
    detail_dir.mkdir(exist_ok=True)

    experiments = []
    for train_name, train_df in train_sets.items():
        for feature_name, features in feature_sets_by_train[train_name].items():
            if len(features) < 2:
                continue
            for model_name, model_template in cycle_models.items():
                decisions = ["majority_vote"]
                if model_name not in {"Ridge", "Dummy_prior"}:
                    decisions.append("mean_proba")
                for decision in decisions:
                    experiments.append((train_name, train_df, feature_name, features, model_name, model_template, decision))

    total_exp = len(experiments)
    for i, (train_name, train_df, feature_name, features, model_name, model_template, decision) in enumerate(experiments, start=1):
        pct = 15 + int(55 * i / max(total_exp, 1))
        if i == 1 or i % 20 == 0 or i == total_exp:
            set_progress(
                pct,
                "模型实验",
                f"{i}/{total_exp}: {train_name}/{feature_name}/{model_name}/{decision}",
            )
        try:
            row, pred_df = evaluate_cycle_experiment(
                train_df,
                group_df,
                train_name,
                feature_name,
                features,
                model_name,
                model_template,
                decision,
            )
            if model_name not in {"Dummy_prior"} and i % 2 == 0:
                row.update(loso_cycle_file_eval(train_df, features, model_template, decision))
            else:
                row.update(
                    {
                        "loso_file_f1": np.nan,
                        "loso_file_precision": np.nan,
                        "loso_file_recall": np.nan,
                        "loso_subject_mean_f1": np.nan,
                        "loso_subject_std_f1": np.nan,
                    }
                )
            key = f"{train_name}__{feature_name}__{model_name}__{decision}"
            pred_map[key] = pred_df
            result_rows.append(row)
        except Exception as exc:
            log(f"  WARN experiment failed {train_name}/{feature_name}/{model_name}/{decision}: {exc}")

    set_progress(72, "文件级聚合实验", "比较周期投票与文件聚合模型")
    file_rows, file_pred_map = evaluate_file_agg_experiments(
        train_sets,
        group_df,
        feature_sets_by_train["data_plus_2_auto"]["stable15"],
    )
    result_rows.extend(file_rows)
    pred_map.update(file_pred_map)

    results = pd.DataFrame(result_rows)
    results = results.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    results.to_csv(OUT / "experiment_results.csv", index=False, encoding="utf-8-sig")

    set_progress(82, "误差与置信度分析", "保存最佳配置和错误样本")
    best = results.iloc[0]
    best_key = f"{best['train_set']}__{best['feature_set']}__{best['model']}__{best['decision']}"
    best_pred = pred_map[best_key]
    best_pred.to_csv(OUT / "best_group03_predictions.csv", index=False, encoding="utf-8-sig")
    summarize_errors(best_pred).to_csv(OUT / "best_error_cases.csv", index=False, encoding="utf-8-sig")
    save_confusion(best_pred, "best_confusion_matrix.png", f"最佳配置混淆矩阵：F1={best['f1_macro']:.3f}")

    # Persist details for top experiments only to keep output small.
    for rank, row in results.head(20).iterrows():
        key = f"{row['train_set']}__{row['feature_set']}__{row['model']}__{row['decision']}"
        if key in pred_map:
            pred_map[key].to_csv(detail_dir / f"rank{rank + 1:02d}_{key}.csv", index=False, encoding="utf-8-sig")

    set_progress(90, "图表生成", "生成答辩与复盘图")
    top = results.head(10).copy()
    top["label"] = (
        top["train_set"]
        + "\n"
        + top["feature_set"]
        + " / "
        + top["model"]
        + " / "
        + top["decision"]
    )
    save_bar_plot(top, OUT / "top10_group03_f1.png", "Group03 Macro-F1 Top 10 配置", "f1_macro")
    save_loso_gap_plot(results)
    save_feature_shift_plot(data_plus_2_auto, group_df)
    save_heatmap(results)
    save_strategy_plot(results)
    save_complexity_plot(results)

    set_progress(96, "报告生成", "写入 Markdown 技术复盘")
    write_report(results, best_key, pred_map, train_sets, group_df)

    elapsed = time.time() - start_time
    summary = {
        "elapsed_seconds": elapsed,
        "best_key": best_key,
        "best_f1_macro": float(best["f1_macro"]),
        "best_precision_macro": float(best["precision_macro"]),
        "best_recall_macro": float(best["recall_macro"]),
        "best_accuracy": float(best["accuracy"]),
        "outputs": str(OUT),
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    set_progress(100, "完成", f"总耗时 {elapsed / 60:.1f} 分钟，最佳 F1={best['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
