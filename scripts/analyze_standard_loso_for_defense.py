#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a defense-oriented LOSO analysis for action classification on standard
cycles from the project's own labeled dataset.

Scope:
  - data/labels_v2.csv
  - quality_label == "标准"
  - leave-one-subject-out evaluation
  - model/feature-set comparison, confusion matrix, feature importance
  - Markdown script for the defense narrative
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "standard_loso_assets"
REPORT_DIR = ROOT_DIR / "docs" / "reports"
LABELS_FILE = DATA_DIR / "labels_v2.csv"

sys.path.insert(0, str(ROOT_DIR))

from src.features import extract_cycle_features  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.utils import data_loader  # noqa: E402

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
ACTION_ORDER = ["前平举", "侧平举", "推肩"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_ORDER)}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def parse_subject(filename: str) -> str:
    return Path(filename).stem.split("_")[1]


def load_standard_features() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_FILE)
    labels = labels[labels["quality_label"] == "标准"].copy()
    labels["subject_id"] = labels["filename"].map(parse_subject)

    filtered_cache: dict[str, tuple[np.ndarray, float]] = {}
    rows: list[dict] = []

    for _, label in labels.iterrows():
        filename = label["filename"]
        path = DATA_DIR / filename
        if filename not in filtered_cache:
            raw, fs = data_loader(str(path))
            filtered, _ = preprocess(raw, fs)
            filtered_cache[filename] = (filtered, fs)
        else:
            filtered, fs = filtered_cache[filename]

        start = int(label["start_idx"])
        end = int(label["end_idx"])
        start = max(0, start)
        end = min(len(filtered) - 1, end)
        if end <= start:
            continue

        feat = extract_cycle_features(
            filtered[start : end + 1, 0],
            filtered[start : end + 1, 1],
            fs,
        )
        feat.update(
            {
                "filename": filename,
                "subject_id": label["subject_id"],
                "cycle_id": int(label["cycle_id"]),
                "action_label": label["action_label"],
                "quality_label": label["quality_label"],
                "start_time": float(label["start_time"]),
                "end_time": float(label["end_time"]),
                "duration": float(label["end_time"]) - float(label["start_time"]),
            }
        )
        rows.append(feat)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No standard cycles were loaded.")
    return df


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    meta = {
        "filename",
        "subject_id",
        "cycle_id",
        "action_label",
        "quality_label",
        "start_time",
        "end_time",
        "duration",
    }
    cols = []
    for col in df.columns:
        if col in meta:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def feature_sets(all_cols: list[str]) -> OrderedDict[str, list[str]]:
    basic_bases = ["rms", "mav", "var", "wl", "zc", "ssc", "iemg", "mf", "mdf", "pf"]
    basic = [f"{base}_{ch}" for base in basic_bases for ch in ("ch1", "ch2")]
    basic += [
        "ratio_rms",
        "diff_rms",
        "rms_ratio_ch1",
        "rms_ratio_ch2",
        "corr_coef",
        "peak_time_ch1",
        "peak_time_ch2",
        "activation_time_diff",
    ]

    selected_15 = [
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

    synergy_keywords = (
        "ratio",
        "_dom",
        "corr",
        "peak_time",
        "activation_time",
        "lead",
        "lag",
        "rise",
        "fall",
        "coactivation",
        "crest",
        "env_skew",
        "n_peaks",
    )
    synergy = [c for c in all_cols if any(k in c for k in synergy_keywords)]
    # Keep frequency and sign-change features because they are relatively less
    # dependent on absolute amplitude across subjects.
    synergy += [
        c
        for c in all_cols
        if c.startswith(("mf_", "mdf_", "pf_", "zc_", "ssc_")) and c not in synergy
    ]

    clean = OrderedDict()
    clean["文献基础特征"] = [c for c in basic if c in all_cols]
    clean["协同/比值特征"] = [c for c in synergy if c in all_cols]
    clean["15个稳健特征"] = [c for c in selected_15 if c in all_cols]
    clean["全部候选特征"] = list(all_cols)
    return clean


def build_models() -> OrderedDict[str, Pipeline]:
    return OrderedDict(
        [
            (
                "LogisticRegression",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
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
                "RidgeClassifier",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                        ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced")),
                    ]
                ),
            ),
            (
                "RBF-SVM",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            SVC(
                                C=1.0,
                                kernel="rbf",
                                gamma="scale",
                                class_weight="balanced",
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "RandomForest",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "clf",
                            RandomForestClassifier(
                                n_estimators=500,
                                min_samples_leaf=2,
                                class_weight="balanced",
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "ExtraTrees",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "clf",
                            ExtraTreesClassifier(
                                n_estimators=500,
                                min_samples_leaf=2,
                                class_weight="balanced",
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "KNN",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
                    ]
                ),
            ),
        ]
    )


def evaluate_loso(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    labels: list[int],
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    y_true_all = []
    y_pred_all = []
    subject_rows = []

    for subject in sorted(np.unique(subjects)):
        train_mask = subjects != subject
        test_mask = subjects == subject
        if test_mask.sum() == 0:
            continue

        fold_model = clone(model)
        fold_model.fit(X[train_mask], y[train_mask])
        pred = fold_model.predict(X[test_mask])
        present_labels = sorted(np.unique(y[test_mask]).tolist())

        subject_rows.append(
            {
                "test_subject": subject,
                "samples": int(test_mask.sum()),
                "present_classes": ",".join(ACTION_ORDER[i] for i in present_labels),
                "precision_macro": precision_score(
                    y[test_mask], pred, average="macro", labels=present_labels, zero_division=0
                ),
                "recall_macro": recall_score(
                    y[test_mask], pred, average="macro", labels=present_labels, zero_division=0
                ),
                "f1_macro": f1_score(
                    y[test_mask], pred, average="macro", labels=present_labels, zero_division=0
                ),
                "accuracy": accuracy_score(y[test_mask], pred),
            }
        )
        y_true_all.extend(y[test_mask])
        y_pred_all.extend(pred)

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)

    metrics = {
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "precision_macro": precision_score(
            y_true_all, y_pred_all, average="macro", labels=labels, zero_division=0
        ),
        "recall_macro": recall_score(
            y_true_all, y_pred_all, average="macro", labels=labels, zero_division=0
        ),
        "f1_macro": f1_score(
            y_true_all, y_pred_all, average="macro", labels=labels, zero_division=0
        ),
        "f1_subject_mean": pd.DataFrame(subject_rows)["f1_macro"].mean(),
        "f1_subject_std": pd.DataFrame(subject_rows)["f1_macro"].std(),
        "f1_subject_min": pd.DataFrame(subject_rows)["f1_macro"].min(),
    }
    return metrics, pd.DataFrame(subject_rows), y_true_all, y_pred_all


def compare_models(df: pd.DataFrame, feature_map: OrderedDict[str, list[str]]):
    y = df["action_label"].map(ACTION_TO_ID).to_numpy()
    subjects = df["subject_id"].to_numpy()
    label_ids = list(range(len(ACTION_ORDER)))
    rows = []
    predictions = {}
    subject_tables = {}

    for feature_name, cols in feature_map.items():
        X = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        for model_name, model in build_models().items():
            metrics, by_subject, y_true, y_pred = evaluate_loso(
                model, X, y, subjects, label_ids
            )
            row = {
                "feature_set": feature_name,
                "n_features": len(cols),
                "model": model_name,
                **metrics,
            }
            rows.append(row)
            key = (feature_name, model_name)
            predictions[key] = (y_true, y_pred)
            subject_tables[key] = by_subject

    comparison = pd.DataFrame(rows).sort_values(
        ["f1_macro", "recall_macro", "precision_macro"], ascending=False
    )
    return comparison, predictions, subject_tables


def loso_permutation_importance(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: list[str],
    labels: list[int],
) -> pd.DataFrame:
    scorer = make_scorer(
        f1_score, average="macro", labels=labels, zero_division=0
    )
    fold_rows = []
    for subject in sorted(np.unique(subjects)):
        train_mask = subjects != subject
        test_mask = subjects == subject
        fold_model = clone(model)
        fold_model.fit(X[train_mask], y[train_mask])
        if len(np.unique(y[test_mask])) < 1:
            continue
        result = permutation_importance(
            fold_model,
            X[test_mask],
            y[test_mask],
            scoring=scorer,
            n_repeats=20,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        for name, mean, std in zip(
            feature_names, result.importances_mean, result.importances_std
        ):
            fold_rows.append(
                {
                    "test_subject": subject,
                    "feature": name,
                    "importance_mean": mean,
                    "importance_std": std,
                }
            )
    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return fold_df
    summary = (
        fold_df.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False)
    )
    return summary


def save_distribution_plot(df: pd.DataFrame) -> None:
    action_counts = df["action_label"].value_counts().reindex(ACTION_ORDER, fill_value=0)
    subject_counts = (
        df.groupby(["subject_id", "action_label"]).size().unstack(fill_value=0)
    )
    subject_counts = subject_counts.reindex(columns=ACTION_ORDER, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.7]})
    colors = ["#d9783d", "#187e74", "#c14c46"]
    axes[0].bar(action_counts.index, action_counts.values, color=colors)
    axes[0].set_title("标准样本动作分布")
    axes[0].set_ylabel("周期数")
    for i, v in enumerate(action_counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=10)

    bottom = np.zeros(len(subject_counts))
    for label, color in zip(ACTION_ORDER, colors):
        vals = subject_counts[label].values
        axes[1].bar(subject_counts.index, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    axes[1].set_title("标准样本按受试者分布")
    axes[1].set_xlabel("受试者")
    axes[1].set_ylabel("周期数")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "standard_data_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(comparison: pd.DataFrame, best_feature_set: str) -> None:
    sub = comparison[comparison["feature_set"] == best_feature_set].copy()
    sub = sub.sort_values("f1_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(sub["model"], sub["f1_macro"], color="#187e74")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("LOSO Macro F1")
    ax.set_title(f"模型比较（特征集：{best_feature_set}）")
    for i, v in enumerate(sub["f1_macro"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison_best_feature_set.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_set_plot(comparison: pd.DataFrame) -> None:
    best_by_feature = (
        comparison.sort_values("f1_macro", ascending=False)
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("f1_macro", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = ["#6c7a89", "#187e74", "#d9783d", "#c14c46"]
    ax.barh(best_by_feature["feature_set"], best_by_feature["f1_macro"], color=colors[: len(best_by_feature)])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("该特征集下最佳 LOSO Macro F1")
    ax.set_title("特征集比较")
    for i, row in enumerate(best_by_feature.itertuples()):
        ax.text(
            row.f1_macro + 0.01,
            i,
            f"{row.f1_macro:.3f} | {row.model}",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_set_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("LOSO 聚合混淆矩阵（标准样本）")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=13,
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "standard_loso_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_subject_plot(by_subject: pd.DataFrame) -> None:
    s = by_subject.sort_values("test_subject")
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    colors = ["#187e74" if v >= 0.7 else "#d9783d" if v >= 0.5 else "#c14c46" for v in s["f1_macro"]]
    ax.bar(s["test_subject"], s["f1_macro"], color=colors)
    ax.axhline(s["f1_macro"].mean(), color="#1c2637", linestyle="--", linewidth=1.2, label=f"均值 {s['f1_macro'].mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("留出受试者")
    ax.set_ylabel("Macro F1")
    ax.set_title("逐受试者 LOSO 表现")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "standard_loso_by_subject.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(importance: pd.DataFrame, top_n: int = 15) -> None:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.barh(top["feature"], top["importance_mean"], color="#187e74")
    ax.set_xlabel("LOSO permutation importance（Macro F1 下降量）")
    ax.set_title(f"Top {top_n} 特征重要性")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "standard_loso_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def rel(path: Path) -> str:
    return path.as_posix()


def write_report(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    best_key: tuple[str, str],
    by_subject: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: pd.DataFrame,
    feature_cols: list[str],
) -> Path:
    best_feature_set, best_model = best_key
    best = comparison[
        (comparison["feature_set"] == best_feature_set) & (comparison["model"] == best_model)
    ].iloc[0]
    class_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(ACTION_ORDER))),
        target_names=ACTION_ORDER,
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ACTION_ORDER))))
    cm_df = pd.DataFrame(cm, index=[f"真 {c}" for c in ACTION_ORDER], columns=[f"预 {c}" for c in ACTION_ORDER])

    action_counts = df["action_label"].value_counts().reindex(ACTION_ORDER, fill_value=0)
    model_table = (
        comparison[comparison["feature_set"] == best_feature_set]
        .sort_values("f1_macro", ascending=False)
        [["model", "precision_macro", "recall_macro", "f1_macro", "accuracy"]]
        .copy()
    )
    feature_table = (
        comparison.sort_values("f1_macro", ascending=False)
        .groupby("feature_set", as_index=False)
        .first()[["feature_set", "n_features", "model", "precision_macro", "recall_macro", "f1_macro"]]
        .sort_values("f1_macro", ascending=False)
    )
    top_features = feature_importance.head(12).copy()

    lines = []
    lines.append("# 分类模型剖析讲稿：只基于自有标准数据的 LOSO 结果\n")
    lines.append("> 展示口径：本部分只分析项目自有数据集；训练/验证样本为 `labels_v2.csv` 中 `quality_label == \"标准\"` 的动作周期；不纳入 Group03 隐藏测试集，也不纳入文件夹 2 的补充数据。\n")
    lines.append("## 讲述主线\n")
    lines.append("这部分我不从“我们试了很多模型”开始讲，而是从验证问题本身开始：如果模型将来要面对一个没参与训练的新同学，它能不能只凭双通道 sEMG 的周期特征识别前平举、侧平举和推肩？因此我们采用 LOSO，也就是每次留出一个受试者，其余受试者训练，最后把所有留出结果合并计算 Precision、Recall、F1 和混淆矩阵。\n")

    lines.append("## 1. 数据口径先固定：只看标准动作，避免质量差异干扰动作分类\n")
    lines.append(f"- 标准周期总数：**{len(df)}**")
    lines.append(f"- 受试者数：**{df['subject_id'].nunique()}**")
    lines.append(f"- 动作分布：前平举 {int(action_counts['前平举'])}，侧平举 {int(action_counts['侧平举'])}，推肩 {int(action_counts['推肩'])}")
    lines.append(f"\n![标准样本分布]({rel(OUTPUT_DIR / 'standard_data_distribution.png')})\n")
    lines.append("讲稿：这里我先把分类问题收窄到“标准动作之间怎么分”。这样做的原因是，如果把不标准动作也放进动作分类，模型可能学到的是动作质量或代偿模式，而不是动作类别本身。标准样本的分布并不是完全均衡，这也是为什么后面统一使用 macro Precision、macro Recall 和 macro F1，而不是只看 accuracy。\n")

    lines.append("## 2. 验证方式选择 LOSO，因为它更接近真实泛化场景\n")
    lines.append("LOSO 的含义是 Leave-One-Subject-Out：每一轮把一个受试者完全留作测试，其余受试者训练。这个验证方式比随机划分更严格，因为随机划分可能让同一个人的不同周期同时出现在训练集和测试集里，指标会偏乐观。我们的目标是验证跨人的动作识别，所以 LOSO 更符合任务风险。\n")
    lines.append(f"\n![逐受试者 LOSO 表现]({rel(OUTPUT_DIR / 'standard_loso_by_subject.png')})\n")
    lines.append("说明：逐受试者柱状图的 F1 只在该受试者实际出现过的动作类别上计算；最终模型对比仍以所有 LOSO 测试折合并后的 macro Precision、Recall、F1 为主指标。\n")

    lines.append("## 3. 模型选择：用同一套 LOSO 口径比较，而不是只看训练集拟合\n")
    lines.append("在当前最优特征集下，不同模型的 LOSO 结果如下：\n")
    lines.append(model_table.to_markdown(index=False, floatfmt=".3f"))
    lines.append(f"\n![模型比较]({rel(OUTPUT_DIR / 'model_comparison_best_feature_set.png')})\n")
    lines.append(
        f"结论：本轮只看自有标准数据时，最优组合是 **{best_model} + {best_feature_set}**，"
        f"LOSO macro Precision = **{best['precision_macro']:.3f}**，"
        f"Recall = **{best['recall_macro']:.3f}**，"
        f"F1 = **{best['f1_macro']:.3f}**。"
    )
    lines.append("模型选择理由不是“模型越复杂越好”，而是看跨受试者留出时能否稳定识别三类动作。树模型能处理非线性边界，线性模型更稳定且可解释，SVM/KNN 对小样本边界敏感；最终选择取决于 LOSO 下的 macro F1 和错误模式，而不是单次随机划分。\n")

    lines.append("## 4. 特征选择：文献基础特征是底座，双通道协同特征解释动作差异\n")
    lines.append("我们比较了四组特征：文献基础特征、协同/比值特征、15 个稳健特征、全部候选特征。每组都用相同 LOSO 口径选出该组下表现最好的模型：\n")
    lines.append(feature_table.to_markdown(index=False, floatfmt=".3f"))
    lines.append(f"\n![特征集比较]({rel(OUTPUT_DIR / 'feature_set_comparison.png')})\n")
    lines.append("讲稿：RMS、MAV、WL、ZC、SSC、MF/MDF 这类特征来自常见 sEMG 分类工作，分别描述激活强度、波形复杂度和频谱结构。但我们的任务只有两个通道，而且动作差异和三角肌前束/中束的相对激活有关，所以不能只看单通道绝对幅值。CH2/CH1 比值、RMS 主导度、通道峰值时序等协同特征能把“哪块肌肉更主导”显式表达出来，这就是特征选择的核心理由。\n")

    lines.append("## 5. 混淆矩阵：分类瓶颈要从哪几类互相混淆里看\n")
    lines.append(cm_df.to_markdown())
    lines.append(f"\n![LOSO 混淆矩阵]({rel(OUTPUT_DIR / 'standard_loso_confusion_matrix.png')})\n")
    lines.append("讲稿：混淆矩阵比单个 F1 更有解释价值。如果侧平举错误少，说明 CH2/CH1 这类中束主导特征是有效的；如果推肩和前平举互相混淆，就说明仅靠当前双通道周期统计还不能完全描述推肩这种复合动作，需要进一步加入周期内形态或更多肌肉通道信息。\n")

    lines.append("## 6. 特征重要性：模型真正依赖的是幅值、比值和主导关系\n")
    lines.append(top_features[["feature", "importance_mean", "importance_std"]].to_markdown(index=False, floatfmt=".4f"))
    lines.append(f"\n![特征重要性]({rel(OUTPUT_DIR / 'standard_loso_feature_importance.png')})\n")
    lines.append("讲稿：这里的特征重要性使用 LOSO 测试折上的 permutation importance：也就是每次打乱一个特征，看 Macro F1 下降多少。越靠前的特征，说明它对跨受试者分类越关键。这个结果可以和生理解释对应起来：三角肌前束和中束的绝对激活强度提供动作负荷信息，而比值/主导度特征提供肌肉协同信息。\n")

    lines.append("## 7. 可以这样收束这一页模型分析\n")
    lines.append("这一部分的结论是：在我们自己的标准数据上，动作分类并不是单纯靠一个黑箱模型完成的。验证方式上，我们用 LOSO 避免同一受试者泄漏；模型选择上，我们用 macro F1 和混淆矩阵判断跨人泛化；特征选择上，我们把常见 EMG 特征作为底座，再加入符合肩部动作生理的双通道协同特征。这个结果说明当前系统已经能在标准动作上建立可解释的分类边界，但推肩这类复合动作仍然是后续改进重点。\n")

    lines.append("## 备用数据：分类报告\n")
    lines.append("```text")
    lines.append(class_report.rstrip())
    lines.append("```\n")

    lines.append("## 参考依据\n")
    lines.append("- SENIAM 关于表面肌电传感器位置和固定方式的建议强调：肌肉对应位置、双极电极、方向与间距会影响 sEMG 采集质量，因此本项目把两个通道分别对应三角肌前束和中束。[SENIAM placement/fixation](https://seniam.org/fixation.htm)")
    lines.append("- sEMG 分类综述中常见特征包括 MAV、RMS、WL、ZC、SSC、IEMG、MF/MDF 等，这支持我们用时域、频域和波形复杂度特征作为基础。[Surface EMG Signal Processing and Classification Techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC3821366/)")
    lines.append("- 时间域 EMG 特征稳定性研究解释了 MAV、ZC、SSC 等特征在模式识别中的定义和作用，因此这些特征适合作为基础候选特征。[Study of stability of time-domain features for EMG pattern recognition](https://jneuroengrehab.biomedcentral.com/articles/10.1186/1743-0003-7-21)")

    report_path = REPORT_DIR / "defense_model_standard_loso_script.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    print("Loading standard labeled cycles...")
    df = load_standard_features()
    all_cols = numeric_feature_columns(df)
    fmap = feature_sets(all_cols)

    # Keep only labels in ACTION_ORDER and preserve label order.
    df = df[df["action_label"].isin(ACTION_ORDER)].copy()
    print(f"Standard cycles: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(df["action_label"].value_counts().reindex(ACTION_ORDER, fill_value=0))

    save_distribution_plot(df)
    comparison, predictions, subject_tables = compare_models(df, fmap)
    comparison.to_csv(OUTPUT_DIR / "standard_loso_model_comparison.csv", index=False, encoding="utf-8-sig")

    best_row = comparison.iloc[0]
    best_key = (best_row["feature_set"], best_row["model"])
    print("Best:", best_key)
    print(best_row)

    y_true, y_pred = predictions[best_key]
    by_subject = subject_tables[best_key]
    by_subject.to_csv(OUTPUT_DIR / "standard_loso_by_subject.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ACTION_ORDER))))
    pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in ACTION_ORDER],
        columns=[f"pred_{c}" for c in ACTION_ORDER],
    ).to_csv(OUTPUT_DIR / "standard_loso_confusion_matrix.csv", encoding="utf-8-sig")

    save_model_comparison_plot(comparison, best_key[0])
    save_feature_set_plot(comparison)
    save_confusion_matrix(cm, ACTION_ORDER)
    save_subject_plot(by_subject)

    best_features = fmap[best_key[0]]
    X_best = df[best_features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y_best = df["action_label"].map(ACTION_TO_ID).to_numpy()
    subjects = df["subject_id"].to_numpy()
    best_model = build_models()[best_key[1]]
    importance = loso_permutation_importance(
        best_model,
        X_best,
        y_best,
        subjects,
        best_features,
        list(range(len(ACTION_ORDER))),
    )
    importance.to_csv(OUTPUT_DIR / "standard_loso_feature_importance.csv", index=False, encoding="utf-8-sig")
    save_feature_importance_plot(importance)

    report_path = write_report(
        df,
        comparison,
        best_key,
        by_subject,
        y_true,
        y_pred,
        importance,
        best_features,
    )
    print(f"Report: {report_path}")
    print(f"Assets: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
