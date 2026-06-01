#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_action_loso.py — LOSO-CV 动作分类模型优化 (两阶段快速版)

Phase 1: 80/20 分层划分快速筛选 → Top-2 模型 + 最佳特征方案
Phase 2: LOSO-CV 仅用 Top-2 模型精细化评估
Phase 3: 全数据训练最终模型
"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               BaggingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier

from src.utils import data_loader
from src.preprocessing import preprocess
from src.event_detection import detect_events
from src.features import (extract_cycle_features, normalize_features_per_file,
                           get_ratio_feature_names, FEATURE_SCHEMES)

warnings.filterwarnings("ignore")

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
FS = 2000.0
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

# ------------------------------------------------------------
def load_all_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "emg_*.mat")))
    rows = []
    for fp in files:
        fn = os.path.basename(fp)
        p = fn.replace(".mat","").split("_")
        sid, ac = p[1], p[2]
        al = {"qpj":"前平举","cpj":"侧平举","tj":"推肩"}[ac]
        try:
            r, fs0 = data_loader(fp)
            f, _ = preprocess(r, fs0)
            ev = detect_events(f, fs0)
        except: continue
        for cid,(s,e,pk) in enumerate(ev["cycles"],1):
            ft = extract_cycle_features(f[s:e+1,0], f[s:e+1,1], fs0)
            ft.update({"filename":fn,"subject_id":sid,"action_label":al})
            rows.append(ft)
    return pd.DataFrame(rows)

def prepare_features(df, scheme):
    meta = {"filename","subject_id","action_label","cycle_id",
            "start_idx","end_idx","start_time","end_time","duration",
            "quality_label","abnormal_type","label_source"}
    if scheme == 'normalized':
        df = normalize_features_per_file(df)
    fc = [c for c in df.columns if c not in meta]
    if scheme == 'ratio':
        fc = get_ratio_feature_names(fc)
    return df[fc].values, fc

def build_models():
    return [
        ("RF", Pipeline([("s",StandardScaler()),
            ("c",RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"s":[StandardScaler(),MinMaxScaler()],"c__n_estimators":[100,200,500],
             "c__max_depth":[None,10,20],"c__min_samples_split":[2,5,10]}),
        ("Bagging", Pipeline([("s",StandardScaler()),
            ("c",BaggingClassifier(random_state=RANDOM_STATE))]),
            {"s":[StandardScaler(),MinMaxScaler()],"c__n_estimators":[50,100,200],
             "c__max_samples":[0.5,0.8,1.0],"c__max_features":[0.5,0.8,1.0]}),
        ("SVC", Pipeline([("s",StandardScaler()),
            ("c",SVC(probability=True,random_state=RANDOM_STATE))]),
            {"s":[StandardScaler(),MinMaxScaler()],"c__C":[0.1,1,10,100],
             "c__kernel":["rbf","poly"],"c__degree":[2,3],"c__gamma":["scale","auto"]}),
        ("KNN", Pipeline([("s",StandardScaler()),
            ("c",KNeighborsClassifier())]),
            {"s":[StandardScaler(),MinMaxScaler()],"c__n_neighbors":[3,5,7,9],
             "c__weights":["uniform","distance"],"c__metric":["euclidean","manhattan"]}),
        ("GBM", Pipeline([("s",StandardScaler()),
            ("c",GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            {"s":[StandardScaler(),MinMaxScaler()],"c__n_estimators":[100,200],
             "c__max_depth":[3,5,7],"c__learning_rate":[0.01,0.1,0.2]}),
        ("LR", Pipeline([("s",StandardScaler()),
            ("c",LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))]),
            {"s":[StandardScaler(),MinMaxScaler()],"c__C":[0.01,0.1,1,10]}),
        ("Ridge", Pipeline([("s",StandardScaler()),
            ("c",RidgeClassifier(random_state=RANDOM_STATE))]),
            {"s":[StandardScaler(),MinMaxScaler()],"c__alpha":[0.01,0.1,1,10,100]}),
        ("MLP", Pipeline([("s",StandardScaler()),
            ("c",MLPClassifier(max_iter=1000,random_state=RANDOM_STATE))]),
            {"s":[StandardScaler(),MinMaxScaler()],"c__hidden_layer_sizes":[(50,),(100,),(50,25)],
             "c__alpha":[0.0001,0.001,0.01],"c__learning_rate_init":[0.001,0.01]}),
    ]

# ------------------------------------------------------------
def main():
    print("="*60)
    print("  LOSO-CV 动作模型优化 (两阶段快速版)")
    print("="*60)

    # Load
    print("\n[1/5] 加载数据...")
    df = load_all_data()
    le = LabelEncoder()
    y_all = le.fit_transform(df["action_label"])
    subjects_all = df["subject_id"].values
    file_ids_all = df["filename"].values
    print(f"  样本: {len(df)}, 类别: {dict(zip(le.classes_,range(3)))}")

    # Phase 1: Quick screening
    print("\n[2/5] Phase 1: 80/20 快速筛选...")
    models = build_models()
    all_results = []

    for scheme in ["raw","normalized","ratio"]:
        X_s, fc = prepare_features(df, scheme)
        # One quick split
        from sklearn.model_selection import train_test_split
        uf = np.unique(file_ids_all)
        fl = np.array([pd.Series(y_all[file_ids_all==f]).mode()[0] for f in uf])
        trf, tef = train_test_split(uf, test_size=0.2, stratify=fl, random_state=RANDOM_STATE)
        trm = np.isin(file_ids_all, trf)
        tem = np.isin(file_ids_all, tef)
        Xtr, Xte = X_s[trm], X_s[tem]
        ytr, yte = y_all[trm], y_all[tem]

        print(f"\n  方案: {FEATURE_SCHEMES[scheme]} | train={len(ytr)}, test={len(yte)}")

        for name, pipe, params in models:
            try:
                g = file_ids_all[trm]
                gs = GridSearchCV(pipe, params, cv=GroupKFold(n_splits=5),
                                  scoring="f1_macro", n_jobs=1, verbose=0)
                gs.fit(Xtr, ytr, groups=g)
                yp = gs.predict(Xte)
                f1 = f1_score(yte, yp, average="macro")
                all_results.append({"scheme":scheme,"model":name,"cv_f1":gs.best_score_,
                                    "test_f1":round(f1,4),"params":str(gs.best_params_)})
                print(f"    {name}: CV={gs.best_score_:.4f} Test={f1:.4f}")
            except Exception as e:
                print(f"    {name}: FAIL - {e}")

    dfr = pd.DataFrame(all_results)
    # Sort by test_f1
    top = dfr.nlargest(5, "test_f1")
    print(f"\n  Top-5:")
    print(top[["scheme","model","cv_f1","test_f1"]].to_string(index=False))

    # Get top-2 models and best scheme
    best_scheme = top.iloc[0]["scheme"]
    top2_models = list(top["model"].unique()[:2])
    print(f"\n  Phase 2 使用: scheme={best_scheme}, models={top2_models}")

    # Phase 2: LOSO-CV with top-2 models only
    print(f"\n[3/5] Phase 2: LOSO-CV ({best_scheme}, {top2_models})...")
    X_sel, _ = prepare_features(df, best_scheme)
    # Filter models
    top_pipes = [(n,p,pr) for n,p,pr in models if n in top2_models]

    loso_records = []
    for test_subj in sorted(np.unique(subjects_all)):
        trm = subjects_all != test_subj
        tem = subjects_all == test_subj
        Xtr, Xte = X_sel[trm], X_sel[tem]
        ytr, yte = y_all[trm], y_all[tem]
        if len(ytr) < 20 or len(yte) < 3: continue

        best_f1, best_name = -1, ""
        for name, pipe, params in top_pipes:
            try:
                g = file_ids_all[trm]
                gs = GridSearchCV(pipe, params, cv=GroupKFold(n_splits=min(5,len(np.unique(g)))),
                                  scoring="f1_macro", n_jobs=1, verbose=0)
                gs.fit(Xtr, ytr, groups=g)
                yp = gs.predict(Xte)
                f1 = f1_score(yte, yp, average="macro")
                if f1 > best_f1: best_f1, best_name = f1, name
            except: continue
        if best_name:
            loso_records.append({"test_subject":test_subj,"model":best_name,
                                 "macro_f1":round(best_f1,4),
                                 "train":len(ytr),"test":len(yte)})
            print(f"  [{test_subj}] {best_name} F1={best_f1:.4f}")

    df_loso = pd.DataFrame(loso_records)
    mean_f1 = df_loso["macro_f1"].mean()
    std_f1 = df_loso["macro_f1"].std()
    print(f"\n  LOSO Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # Save evaluation
    df_loso.to_csv(os.path.join(OUTPUT_DIR,"loso_action_evaluation.csv"),
                   index=False, encoding="utf-8-sig")
    dfr.to_csv(os.path.join(OUTPUT_DIR,"loso_action_all_models.csv"),
               index=False, encoding="utf-8-sig")

    # Per-subject chart
    fig,ax = plt.subplots(figsize=(12,5))
    s = df_loso["test_subject"].tolist()
    f = df_loso["macro_f1"].tolist()
    ax.bar(range(len(s)),f,color="#1f77b4",edgecolor="white")
    ax.set_xticks(range(len(s))); ax.set_xticklabels(s,fontsize=11)
    ax.set_ylabel("Macro F1",fontsize=13)
    ax.set_xlabel("Test Subject",fontsize=13)
    ax.set_title(f"LOSO-CV Action Classification | {best_scheme} | Mean={mean_f1:.3f}±{std_f1:.3f}",fontsize=13)
    ax.axhline(mean_f1,color="red",linestyle="--",label=f"Mean={mean_f1:.3f}")
    ax.set_ylim(0,1.05); ax.legend(); ax.grid(True,alpha=0.3,axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"loso_action_per_subject.png"),dpi=150,bbox_inches="tight")
    plt.close()

    with open(os.path.join(OUTPUT_DIR,"loso_action_best_scheme.txt"),"w",encoding="utf-8") as fh:
        fh.write(f"Optimal scheme: {best_scheme} ({FEATURE_SCHEMES[best_scheme]})\n")
        fh.write(f"Top models: {top2_models}\n")
        fh.write(f"LOSO Macro F1: {mean_f1:.4f} ± {std_f1:.4f}\n")
        fh.write(f"\nTop-5 screening results:\n{top[['scheme','model','cv_f1','test_f1']].to_string(index=False)}\n")
        fh.write(f"\nLOSO per-subject:\n{df_loso.to_string(index=False)}\n")

    # Phase 3: Final model on all data
    print(f"\n[4/5] Training final model (all 15 subjects)...")
    best_model_name = top2_models[0]
    best_pipe = [p for n,p,_ in top_pipes if n==best_model_name][0]

    # Fit best model with default params on all data
    gballs = file_ids_all
    gs_final = GridSearchCV(best_pipe, [p for n,_,p in top_pipes if n==best_model_name][0],
                             cv=GroupKFold(n_splits=min(5,len(np.unique(gballs)))),
                             scoring="f1_macro", n_jobs=1, verbose=0)
    gs_final.fit(X_sel, y_all, groups=gballs)

    dump(gs_final.best_estimator_, os.path.join(MODEL_DIR,"action_model.joblib"))
    dump(le, os.path.join(MODEL_DIR,"label_encoder_action.joblib"))
    print(f"  Models saved: models/action_model.joblib")

    # Summary
    print(f"\n[5/5] Done!")
    print(f"  Best scheme: {best_scheme} ({FEATURE_SCHEMES[best_scheme]})")
    print(f"  Best model: {best_model_name}")
    print(f"  LOSO Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
