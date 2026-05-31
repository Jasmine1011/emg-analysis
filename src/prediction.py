# -*- coding: utf-8 -*-
"""
src/prediction.py — 预测模块

功能:
    1. 加载训练好的动作分类模型 + 质量分类模型
    2. 对上传文件进行周期级预测
    3. 多数投票得到文件级动作类别
    4. 不标准周期规则解释 (基于 CH2/CH1 RMS 比值)

模型路径:
    models/action_model.joblib
    models/quality_model.joblib
    models/label_encoder_action.joblib
    models/label_encoder_quality.joblib

使用方式:
    from src.prediction import predict_action, predict_quality
"""

import os
import numpy as np
from joblib import load


# ------------------------------------------------------------
# 模型路径
# ------------------------------------------------------------
def _get_emg_dir():
    """获取项目根目录的绝对路径"""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(src_dir)


def _model_path(name):
    return os.path.join(_get_emg_dir(), "models", name)


# ------------------------------------------------------------
# 模型加载
# ------------------------------------------------------------
def load_action_model():
    """
    加载动作分类模型和标签编码器.
    返回 (pipeline, label_encoder) 或 (None, None)
    """
    mp = _model_path("action_model.joblib")
    lp = _model_path("label_encoder_action.joblib")
    if not os.path.exists(mp):
        return None, None
    return load(mp), load(lp)


def load_quality_model():
    """
    加载质量分类模型和标签编码器.
    返回 (pipeline, label_encoder) 或 (None, None)
    """
    mp = _model_path("quality_model.joblib")
    lp = _model_path("label_encoder_quality.joblib")
    if not os.path.exists(mp):
        return None, None
    return load(mp), load(lp)


# ------------------------------------------------------------
# 特征提取辅助
# ------------------------------------------------------------
def _extract_features_for_prediction(filtered_data, fs, cycles):
    """
    对所有周期提取特征，返回特征矩阵 X 和每个周期的 RMS 比值.

    Returns
    -------
    X : np.ndarray, shape (n_cycles, n_features)
    ratios : list of float
    """
    from .features import extract_cycle_features

    rows = []
    ratios = []

    for s, e, pk in cycles:
        seg1 = filtered_data[s:e + 1, 0]
        seg2 = filtered_data[s:e + 1, 1]
        feat = extract_cycle_features(seg1, seg2, fs)

        # 记录 RMS 比值用于规则解释
        rms1 = float(np.sqrt(np.mean(seg1 ** 2)))
        rms2 = float(np.sqrt(np.mean(seg2 ** 2)))
        ratio = rms2 / rms1 if rms1 > 1e-12 else float("inf")
        ratios.append(ratio)

        rows.append(feat)

    # 构建特征矩阵 (对齐训练时的特征列顺序)
    import pandas as pd
    df = pd.DataFrame(rows)
    exclude = ["cycle_id", "start_idx", "end_idx", "start_time", "end_time",
               "filename", "action_label", "quality_label"]
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values

    return X, ratios


# ------------------------------------------------------------
# 动作分类预测
# ------------------------------------------------------------
def predict_action(filtered_data, fs, cycles):
    """
    对检测到的周期进行动作分类预测，并通过多数投票确定文件级动作.

    Parameters
    ----------
    filtered_data : np.ndarray, shape (n_samples, 2)
    fs : float
    cycles : list of tuple (start_idx, end_idx, peak_idx)

    Returns
    -------
    result : dict
        {
            'overall_action': str,         # 整体动作类别
            'overall_confidence': float,   # 多数投票比例
            'cycle_results': list of dict, # 逐周期预测
        }
    """
    model, le = load_action_model()
    if model is None:
        return {"overall_action": "模型未训练", "cycle_results": []}

    X, ratios = _extract_features_for_prediction(filtered_data, fs, cycles)
    if len(X) == 0:
        return {"overall_action": "无周期", "cycle_results": []}

    # 预测
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    else:
        y_proba = None

    # 转换为标签名
    pred_labels = le.inverse_transform(y_pred)
    class_names = le.classes_

    # 逐周期结果
    cycle_results = []
    for i, (s, e, pk) in enumerate(cycles):
        dur = (e - s) / fs
        result_entry = {
            "cycle_id": i + 1,
            "start_time": round(s / fs, 2),
            "end_time": round(e / fs, 2),
            "duration": round(dur, 2),
            "prediction": pred_labels[i],
            "ratio": round(ratios[i], 3),
        }
        if y_proba is not None:
            conf = float(np.max(y_proba[i]))
            result_entry["confidence"] = round(conf, 4)
            # 各类别概率
            result_entry["probabilities"] = {
                class_names[j]: round(float(y_proba[i, j]), 4)
                for j in range(len(class_names))
            }
        else:
            result_entry["confidence"] = 1.0

        cycle_results.append(result_entry)

    # 多数投票
    from collections import Counter
    vote_counts = Counter(pred_labels)
    overall = vote_counts.most_common(1)[0][0]
    overall_conf = vote_counts[overall] / len(pred_labels)

    # 标记不一致周期
    for r in cycle_results:
        r["consistent"] = (r["prediction"] == overall)

    return {
        "overall_action": overall,
        "overall_confidence": round(overall_conf, 3),
        "vote_counts": dict(vote_counts),
        "cycle_results": cycle_results,
    }


# ------------------------------------------------------------
# 质量分类预测
# ------------------------------------------------------------
def predict_quality(filtered_data, fs, cycles, action_label=None):
    """
    周期级质量预测 + 不标准周期规则解释.

    Parameters
    ----------
    filtered_data : np.ndarray
    fs : float
    cycles : list of tuple
    action_label : str or None
        整体动作类别 (可选, 用于规则解释).

    Returns
    -------
    result : dict
    """
    model, le = load_quality_model()
    X, ratios = _extract_features_for_prediction(filtered_data, fs, cycles)

    if model is None or len(X) == 0:
        return {"overall_summary": "模型未训练", "cycle_results": []}

    y_pred = model.predict(X)
    pred_labels = le.inverse_transform(y_pred)

    cycle_results = []
    standard_count = 0
    nonstandard_count = 0

    for i, (s, e, pk) in enumerate(cycles):
        dur = (e - s) / fs
        ml_quality = pred_labels[i]
        ratio = ratios[i]

        # 规则解释 (当 ML 预测为不标准时)
        explanation = ""
        abnormal_type = ""
        if ml_quality == "不标准":
            nonstandard_count += 1
            abnormal_type, explanation = explain_abnormality(
                action_label or "", ratio)
        else:
            standard_count += 1

        cycle_results.append({
            "cycle_id": i + 1,
            "start_time": round(s / fs, 2),
            "end_time": round(e / fs, 2),
            "duration": round(dur, 2),
            "ratio": round(ratio, 3),
            "quality": ml_quality,
            "abnormal_type": abnormal_type,
            "explanation": explanation,
        })

    return {
        "standard_count": standard_count,
        "nonstandard_count": nonstandard_count,
        "overall_summary": f"{standard_count} 个标准 / {nonstandard_count} 个不标准",
        "cycle_results": cycle_results,
    }


# ------------------------------------------------------------
# 规则解释
# ------------------------------------------------------------
def explain_abnormality(action_label, ratio):
    """
    基于 CH2/CH1 RMS 比值生成异常解释文本.

    规则 (沿用 quality_rules.py):
        前平举 (qpj): ratio < 0.80 标准, ratio ≥ 0.80 不标准
            ratio > 1.0 → 疑似中束代偿
            0.80 ≤ ratio ≤ 1.0 → 其他不标准
        侧平举 (cpj): ratio > 1.30 标准, ratio ≤ 1.30 不标准
            ratio < 1.0 → 疑似前束代偿
            1.0 ≤ ratio ≤ 1.30 → 其他不标准
        推肩 (tj): ratio < 0.85 标准, ratio ≥ 0.85 不标准
            ratio > 1.0 → 疑似中束代偿
            0.85 ≤ ratio ≤ 1.0 → 其他不标准

    Parameters
    ----------
    action_label : str
        动作类别 (前平举/侧平举/推肩).
    ratio : float
        CH2_RMS / CH1_RMS 比值.

    Returns
    -------
    abnormal_type : str
    explanation : str
    """
    if action_label in ("前平举", "推肩"):
        if ratio > 1.0:
            abnormal_type = "疑似中束代偿"
            explanation = (
                f"该周期的中束/前束 RMS 比值 ({ratio:.2f}) > 1.0，"
                f"中束激活反超前束，说明动作可能存在向外展方向偏移或三角肌中束代偿。"
            )
        elif ratio >= 0.80:
            abnormal_type = "其他不标准"
            explanation = (
                f"该周期的中束/前束 RMS 比值 ({ratio:.2f}) 高于 {action_label} 标准范围 "
                f"(预期 < 0.{'80' if action_label == '前平举' else '85'})，"
                f"但未构成完全代偿，可能存在动作执行不稳定或其他异常。"
            )
        else:
            # ratio 正常但 ML 判定不标准 → 可能是其他特征异常
            abnormal_type = "其他不标准"
            explanation = "该周期的 RMS 比值正常，但其他特征模式异常，可能存在动作执行不稳定。"
    elif action_label == "侧平举":
        if ratio < 1.0:
            abnormal_type = "疑似前束代偿"
            explanation = (
                f"该周期的中束/前束 RMS 比值 ({ratio:.2f}) < 1.0，"
                f"前束激活反超中束，说明手臂可能前摆，动作不符合外侧轨迹。"
            )
        elif ratio <= 1.30:
            abnormal_type = "其他不标准"
            explanation = (
                f"该周期的中束/前束 RMS 比值 ({ratio:.2f}) 低于侧平举标准范围 "
                f"(预期 > 1.30)，但未构成完全代偿，可能存在动作执行不稳定或其他异常。"
            )
        else:
            abnormal_type = "其他不标准"
            explanation = "该周期的 RMS 比值正常，但其他特征模式异常，可能存在动作执行不稳定。"
    else:
        # 未知动作类型
        abnormal_type = "其他不标准"
        explanation = f"中束/前束 RMS 比值 = {ratio:.2f}"

    return abnormal_type, explanation
