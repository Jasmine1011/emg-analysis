#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_action_model.py — 动作类型三分类模型训练脚本

功能:
    1. 加载全部 65 个 .mat 文件，预处理 + 事件检测 + 特征提取
    2. 候选模型 GridSearchCV(cv=5) 超参数搜索比较
    3. 选择最优模型，评估并保存

候选模型:
    RandomForest, BaggingClassifier, SVC(poly/rbf), KNN, XGBoost,
    LightGBM, GradientBoosting, LogisticRegression, Ridge, MLP

输出:
    emg/models/action_model.joblib
    emg/models/scaler_action.joblib
    emg/outputs/action_*.csv / .png

使用方式:
    cd f:/Project_final
    D:/Anaconda_202410/python.exe train_action_model.py
"""

import os
import sys
import glob
import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.model_selection import GridSearchCV, GroupKFold, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier

# XGBoost / LightGBM 可选
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import data_loader
from src.preprocessing import preprocess
from src.event_detection import detect_events
from src.features import extract_cycle_features

# ------------------------------------------------------------
# 全局配置
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 2000.0
RANDOM_STATE = 42
CV_FOLDS = 5

# 特征列名 (仅特征，不含标签)
FEATURE_COLS_CH = [
    "rms", "mav", "var", "wl", "zc", "ssc", "iemg",
    "mf", "mdf", "pf",
]
FEATURE_COLS_CROSS = [
    "ratio_rms", "diff_rms", "rms_ratio_ch1", "rms_ratio_ch2",
    "corr_coef", "peak_time_ch1", "peak_time_ch2", "activation_time_diff",
]


# ------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------
def load_all_data():
    """加载全部 65 个文件的特征，返回 X (DataFrame), y (Series), file_ids (list)"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    print(f"加载 {len(files)} 个文件...")

    all_rows = []
    file_labels = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        # 从文件名解析动作标签
        parts = fname.replace(".mat", "").split("_")
        action_code = parts[2]  # qpj / cpj / tj
        if action_code == "qpj":
            action_label = "前平举"
        elif action_code == "cpj":
            action_label = "侧平举"
        else:
            action_label = "推肩"

        file_labels[fname] = action_label

        try:
            raw, fs = data_loader(fpath)
            filtered, _ = preprocess(raw, fs)
            result = detect_events(filtered, fs)
        except Exception:
            continue

        for cid, (s, e, pk) in enumerate(result["cycles"], start=1):
            seg1 = filtered[s:e + 1, 0]
            seg2 = filtered[s:e + 1, 1]
            feat = extract_cycle_features(seg1, seg2, fs)
            feat["filename"] = fname
            feat["action_label"] = action_label
            all_rows.append(feat)

    df = pd.DataFrame(all_rows)
    print(f"  总样本数: {len(df)}")
    print(f"  类别分布: {df['action_label'].value_counts().to_dict()}")

    # 提取特征列
    feature_names = [c for c in df.columns if c not in
                     ("filename", "action_label", "cycle_id",
                      "start_idx", "end_idx", "start_time", "end_time")]
    X = df[feature_names]
    y = df["action_label"]
    file_ids = df["filename"].values

    return X, y, file_ids, feature_names


# ------------------------------------------------------------
# 按文件分层划分
# ------------------------------------------------------------
def stratified_file_split(X, y, file_ids, test_size=0.20):
    """
    按文件分层划分训练/测试集。
    策略：对每个文件聚合其多数类别 => 按文件级类别分层。
    """
    unique_files = np.unique(file_ids)
    file_to_label = {}
    for f in unique_files:
        mask = file_ids == f
        labels = y[mask]
        # 取众数
        file_to_label[f] = pd.Series(labels).mode()[0]

    file_labels_arr = np.array([file_to_label[f] for f in unique_files])

    from sklearn.model_selection import train_test_split
    train_files, test_files = train_test_split(
        unique_files, test_size=test_size,
        stratify=file_labels_arr,
        random_state=RANDOM_STATE,
    )

    train_mask = np.isin(file_ids, train_files)
    test_mask = np.isin(file_ids, test_files)

    print(f"训练集文件: {len(train_files)}, 样本: {train_mask.sum()}")
    print(f"测试集文件: {len(test_files)}, 样本: {test_mask.sum()}")
    return train_mask, test_mask


# ------------------------------------------------------------
# 候选模型定义
# ------------------------------------------------------------
def build_candidates():
    """构建候选模型 Pipeline + 超参网格列表"""
    candidates = []

    # 1. Random Forest
    candidates.append({
        "name": "RandomForest",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
        "params": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "clf__n_estimators": [100, 200, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
        },
    })

    # 2. Bagging Tree
    candidates.append({
        "name": "BaggingTree",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", BaggingClassifier(random_state=RANDOM_STATE)),
        ]),
        "params": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "clf__n_estimators": [50, 100, 200],
            "clf__max_samples": [0.5, 0.8, 1.0],
            "clf__max_features": [0.5, 0.8, 1.0],
        },
    })

    # 3. SVC
    candidates.append({
        "name": "SVC",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
        "params": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "clf__C": [0.1, 1, 10, 100],
            "clf__kernel": ["rbf", "poly"],
            "clf__degree": [2, 3],
            "clf__gamma": ["scale", "auto"],
        },
    })

    # 4. KNN
    candidates.append({
        "name": "KNN",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]),
        "params": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    })

    # 5. GradientBoosting
    candidates.append({
        "name": "GradientBoosting",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        "params": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.2],
        },
    })

    # 6. LogisticRegression
    candidates.append({
        "name": "LogisticRegression",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "params": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "clf__C": [0.01, 0.1, 1, 10],
        },
    })

    # 7. RidgeClassifier
    candidates.append({
        "name": "RidgeClassifier",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(random_state=RANDOM_STATE)),
        ]),
        "params": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "clf__alpha": [0.01, 0.1, 1, 10, 100],
        },
    })

    # 8. MLP
    candidates.append({
        "name": "MLP",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "params": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 25)],
            "clf__alpha": [0.0001, 0.001, 0.01],
            "clf__learning_rate_init": [0.001, 0.01],
        },
    })

    # 9. XGBoost (optional)
    if HAS_XGB:
        candidates.append({
            "name": "XGBoost",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss")),
            ]),
            "params": {
                "scaler": [StandardScaler(), MinMaxScaler()],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.1, 0.2],
            },
        })

    # 10. LightGBM (optional)
    if HAS_LGB:
        candidates.append({
            "name": "LightGBM",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)),
            ]),
            "params": {
                "scaler": [StandardScaler(), MinMaxScaler()],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.1, 0.2],
            },
        })

    return candidates


# ------------------------------------------------------------
# 绘图辅助
# ------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(cm, class_names, title, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("预测标签", fontsize=13)
    ax.set_ylabel("真实标签", fontsize=13)
    ax.set_title(title, fontsize=14)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  混淆矩阵已保存: {save_path}")


def plot_feature_importance(model, feature_names, title, save_path, top_n=20):
    """绘制特征重要性排名"""
    # 尝试获取特征重要性
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        print("  模型不支持特征重要性可视化")
        return

    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("特征重要性", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  特征重要性已保存: {save_path}")


def plot_learning_curve(model, X, y, title, save_path):
    """绘制学习曲线"""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="f1_macro", random_state=RANDOM_STATE,
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="训练集 F1")
    ax.plot(train_sizes, test_mean, "o-", color="orange", label="交叉验证 F1")
    ax.set_xlabel("训练样本数", fontsize=12)
    ax.set_ylabel("F1 (macro)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  学习曲线已保存: {save_path}")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    print("=" * 64)
    print("  动作类型三分类模型训练 (GridSearchCV, cv=5)")
    print("=" * 64)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    X, y, file_ids, feature_names = load_all_data()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  类别编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 2. 划分训练/测试集
    print("\n[2/5] 按文件分层划分训练/测试集 (80/20)...")
    train_mask, test_mask = stratified_file_split(X, y_encoded, file_ids, test_size=0.20)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]
    # 用于 GroupKFold 的分组：同一文件的所有周期必须在同一折
    groups_train = file_ids[train_mask]
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 3. 候选模型搜索
    print("\n[3/5] GridSearchCV 模型搜索 (GroupKFold cv=5, 按文件分组)...")
    candidates = build_candidates()
    results = []

    for cand in candidates:
        name = cand["name"]
        print(f"\n  >>> {name} ...")
        try:
            gs = GridSearchCV(
                cand["pipeline"], cand["params"],
                cv=GroupKFold(n_splits=CV_FOLDS),
                scoring="f1_macro", n_jobs=-1, verbose=0,
            )
            gs.fit(X_train, y_train, groups=groups_train)

            # 测试集评估
            y_pred = gs.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            prec = precision_score(y_test, y_pred, average="macro")
            rec = recall_score(y_test, y_pred, average="macro")

            results.append({
                "model": name,
                "best_params": str(gs.best_params_),
                "cv_mean_f1": round(gs.best_score_, 4),
                "test_accuracy": round(acc, 4),
                "test_f1_macro": round(f1, 4),
                "test_precision": round(prec, 4),
                "test_recall": round(rec, 4),
                "best_estimator": gs.best_estimator_,
            })
            print(f"    CV F1(macro)={gs.best_score_:.4f}, Test F1={f1:.4f}, Test Acc={acc:.4f}")
        except Exception as exc:
            print(f"    !! 失败: {exc}")

    if not results:
        print("错误：所有模型均训练失败！")
        sys.exit(1)

    # 4. 选择最优模型
    print("\n[4/5] 模型比较与选择...")
    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k != "best_estimator"}
        for r in results
    ])
    df_results = df_results.sort_values("test_f1_macro", ascending=False)
    print(df_results.to_string(index=False))
    df_results.to_csv(os.path.join(OUTPUT_DIR, "action_model_comparison.csv"),
                      index=False, encoding="utf-8-sig")
    print(f"  对比表已保存: {OUTPUT_DIR}/action_model_comparison.csv")

    # 按 test_f1_macro 排序 results 列表
    results.sort(key=lambda r: r["test_f1_macro"], reverse=True)
    best = results[0]
    best_model = best["best_estimator"]
    print(f"\n  最优模型: {best['model']}")
    print(f"  测试 F1(macro): {best['test_f1_macro']:.4f}")
    print(f"  测试准确率: {best['test_accuracy']:.4f}")

    # 5. 评估与保存
    print("\n[5/5] 评估图表与模型保存...")

    # 混淆矩阵
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plot_confusion_matrix(cm, le.classes_,
                          f"动作分类混淆矩阵 ({best['model']})",
                          os.path.join(OUTPUT_DIR, "action_confusion_matrix.png"))

    # 特征重要性
    final_clf = best_model.named_steps["clf"]
    plot_feature_importance(final_clf, feature_names,
                            f"特征重要性排名 ({best['model']})",
                            os.path.join(OUTPUT_DIR, "action_feature_importance.png"))

    # 学习曲线
    plot_learning_curve(best_model, X_train, y_train,
                        f"学习曲线 ({best['model']})",
                        os.path.join(OUTPUT_DIR, "action_learning_curve.png"))

    # 分类报告
    report = classification_report(y_test, y_pred_best, target_names=le.classes_)
    print("\n分类报告:\n" + report)
    with open(os.path.join(OUTPUT_DIR, "action_classification_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(report)

    # 保存模型
    model_path = os.path.join(MODEL_DIR, "action_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler_action.joblib")
    dump(best_model, model_path)
    dump(le, os.path.join(MODEL_DIR, "label_encoder_action.joblib"))
    print(f"\n  模型已保存: {model_path}")

    print("\n" + "=" * 64)
    print("  动作模型训练完成！")
    print(f"  最优模型: {best['model']} (Test F1={best['test_f1_macro']:.4f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
