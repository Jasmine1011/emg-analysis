# -*- coding: utf-8 -*-
"""
app.py — 双通道表面肌电信号处理与分析平台 v3.0

一键式流程: 上传 .mat → 开始分析 → 自动完成全管线 → 结果展示

Tab 结构:
    Tab 1: 📊 原始信号
    Tab 2: 🔧 预处理结果 (自选对比视图)
    Tab 3: 🎯 活动检测 (周期表 + 包络图)
    Tab 4: 📈 特征展示 (自选曲线 + 表格)
    Tab 5: 🧠 分类与质量
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import data_loader
from src.visualization_signal import plot_data, plot_fft_spectrum
from app.style import CUSTOM_CSS

from src.preprocessing import preprocess
from src.event_detection import detect_events
from src.features import compute_feature_curves, extract_all_cycles, load_labels, DEFAULT_SELECTED_FEATURES
from src.visualization import (
    plot_feature_curves as viz_feature_curves,
    plot_envelope_with_cycles,
)
from src.prediction import predict_action, predict_quality

# ------------------------------------------------------------
# 页面配置
# ------------------------------------------------------------
st.set_page_config(
    page_title="EMG 信号分析平台",
    page_icon="💪",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# 初始化 session_state
# ------------------------------------------------------------
DEFAULTS = {
    "analysis_done": False,
    "raw_data": None,
    "fs": None,
    "fname": None,
    "filtered_data": None,
    "events": None,
    "curves": None,
    "features_df": None,
    "action_result": None,
    "quality_result": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------
# 侧边栏
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 数据上传")
    uploaded_file = st.file_uploader("选择 .mat 文件", type=["mat"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        raw_data, fs = data_loader(tmp_path)
        fname = uploaded_file.name
        os.unlink(tmp_path)

        st.session_state.raw_data = raw_data
        st.session_state.fs = fs
        st.session_state.fname = fname

        dur_s = raw_data.shape[0] / fs
        st.success(f"**{fname}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("采样率", f"{fs} Hz")
        c2.metric("样本数", f"{raw_data.shape[0]:,}")
        c3.metric("时长", f"{dur_s:.1f} s")

    st.markdown("---")

    # ---- 高级设置 (折叠) ----
    with st.expander("⚙️ 高级设置", expanded=False):
        notch_freq = st.number_input("工频频率 (Hz)", value=50.0, step=10.0)
        fc_high = st.number_input("高通截止 (Hz)", value=20.0, step=5.0)
        fc_low = st.number_input("低通截止 (Hz)", value=450.0, step=10.0)
        smooth_win = st.slider("平滑窗口 (s)", 0.05, 1.0, 0.300, 0.05)
        min_dur = st.slider("最小周期 (s)", 0.5, 3.0, 1.5, 0.1)
        max_dur = st.slider("最大周期 (s)", 4.0, 15.0, 8.0, 0.5)
        target_k = st.slider("预期周期数", 2, 8, 5)
        st.caption("活动/静息检测")
        activity_sigma = st.slider("静息阈值倍数", 1.0, 8.0, 3.0, 0.5,
                                   help="越大越不敏感。基线 + 阈值倍数×噪声标准差 = 活动阈值")

    # ---- 一键分析按钮 ----
    st.markdown("---")
    btn_disabled = (uploaded_file is None)
    if st.button("🚀 开始分析", use_container_width=True, type="primary",
                 disabled=btn_disabled):
        # 保存参数
        st.session_state._params = dict(
            notch_freq=notch_freq, fc_high=fc_high, fc_low=fc_low,
            smooth_win=smooth_win, min_dur=min_dur, max_dur=max_dur,
            target_k=target_k, activity_sigma=activity_sigma,
        )
        st.session_state.analysis_done = False  # 触发重跑
        st.rerun()

# ------------------------------------------------------------
# 主界面标题
# ------------------------------------------------------------
st.title("💪 双通道表面肌电信号处理与分析")
st.caption(
    "CH1: 三角肌前束 | CH2: 三角肌中束 | "
    "前平举 · 侧平举 · 推肩 | "
    "⚠️ 质量判断为辅助参考"
)

# ------------------------------------------------------------
# 一键分析执行
# ------------------------------------------------------------
def run_analysis():
    """执行全部分析管线，更新 session_state"""
    raw = st.session_state.raw_data
    fs_val = st.session_state.fs
    p = st.session_state._params

    progress_text = "分析进行中..."
    progress_bar = st.progress(0, text=progress_text)
    status_container = st.empty()

    steps = ["预处理", "事件检测", "特征提取", "动作分类", "质量评估"]
    total = len(steps)

    try:
        # Step 1: 预处理
        progress_bar.progress(1 / total, text=f"🔧 {steps[0]}中...")
        filtered, _ = preprocess(
            raw, fs_val,
            apply_dc_removal=True,
            apply_amplitude_scaling=True,
            apply_notch=True,
            notch_freq=p["notch_freq"],
            fc_high=p["fc_high"],
            fc_low=p["fc_low"],
        )
        st.session_state.filtered_data = filtered

        # Step 2: 事件检测
        progress_bar.progress(2 / total, text=f"🎯 {steps[1]}中...")
        events = detect_events(
            filtered, fs_val,
            smooth_window=p["smooth_win"],
            min_duration=p["min_dur"],
            max_duration=p["max_dur"],
            target_k=p["target_k"],
            activity_sigma=p["activity_sigma"],
        )
        st.session_state.events = events

        # Step 3: 特征提取
        progress_bar.progress(3 / total, text=f"📈 {steps[2]}中...")
        curves = compute_feature_curves(filtered, fs_val)
        st.session_state.curves = curves

        labels_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "labels.csv",
        )
        labels_df = load_labels(labels_path)
        features_df = extract_all_cycles(
            filtered, fs_val, events["cycles"],
            filename=st.session_state.fname,
            labels_df=labels_df,
        )
        st.session_state.features_df = features_df

        # Step 4: 动作分类
        progress_bar.progress(4 / total, text=f"🧠 {steps[3]}中...")
        action_result = predict_action(filtered, fs_val, events["cycles"])
        st.session_state.action_result = action_result

        # Step 5: 质量评估
        progress_bar.progress(5 / total, text=f"⚙️ {steps[4]}中...")
        quality_result = predict_quality(
            filtered, fs_val, events["cycles"],
            action_label=action_result.get("overall_action", ""),
        )
        st.session_state.quality_result = quality_result

        # 完成
        progress_bar.progress(1.0, text="✅ 分析完成！")
        st.session_state.analysis_done = True

        n_cycles = events["count"]
        action = action_result.get("overall_action", "N/A")
        status_container.success(
            f"🎉 分析完成 | 检测到 **{n_cycles}** 个周期 | 整体动作: **{action}**"
        )

    except Exception as exc:
        progress_bar.progress(1.0, text="❌ 分析失败")
        status_container.error(f"分析出错: {exc}")
        st.session_state.analysis_done = False


# ---- 触发分析 (仅当用户点击了「开始分析」后) ----
if (uploaded_file is not None
        and not st.session_state.analysis_done
        and '_params' in st.session_state):
    run_analysis()

# ------------------------------------------------------------
# Tab 结构
# ------------------------------------------------------------
if not st.session_state.analysis_done:
    if uploaded_file is None:
        st.info("👈 请上传 .mat 文件并点击「开始分析」")
    else:
        st.info("👈 请点击「开始分析」按钮")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 原始信号", "🔧 预处理结果", "🎯 活动检测",
         "📈 特征展示", "🧠 分类与质量"]
    )

    raw = st.session_state.raw_data
    fs_val = st.session_state.fs
    filtered = st.session_state.filtered_data
    events = st.session_state.events
    curves = st.session_state.curves
    features_df = st.session_state.features_df
    action_result = st.session_state.action_result
    quality_result = st.session_state.quality_result

    # ============================================================
    # Tab 1: 原始信号
    # ============================================================
    with tab1:
        st.subheader("📊 原始 EMG 信号")
        st.markdown(f"**文件**: {st.session_state.fname} | "
                    f"采样率: {fs_val} Hz | 时长: {raw.shape[0]/fs_val:.1f}s")

        st.markdown("#### 双通道时域波形")
        fig = plot_data((raw, fs_val), "原始信号", channels=2)
        st.pyplot(fig)

        st.markdown("#### 幅度谱 (0–500 Hz, dB)")
        fig = plot_fft_spectrum((raw, fs_val), freq_range=(0, 500), scale="dB")
        st.pyplot(fig)

    # ============================================================
    # Tab 2: 预处理结果 (自选对比)
    # ============================================================
    with tab2:
        st.subheader("🔧 预处理结果")
        st.caption("选择要查看的对比视图：")

        view_opts = st.multiselect(
            "对比视图",
            options=["ch1_time", "ch1_freq", "ch2_time", "ch2_freq"],
            default=["ch1_time", "ch2_time"],
            format_func=lambda x: {
                "ch1_time": "CH1 时域对比 (原始 vs 处理后)",
                "ch1_freq": "CH1 频谱对比 (原始 vs 处理后)",
                "ch2_time": "CH2 时域对比 (原始 vs 处理后)",
                "ch2_freq": "CH2 频谱对比 (原始 vs 处理后)",
            }[x],
            label_visibility="collapsed",
        )

        for view in view_opts:
            if view == "ch1_time":
                st.markdown("#### CH1 三角肌前束 — 时域对比")
                c1, c2 = st.columns(2)
                with c1:
                    fig = plot_data((raw[:, 0:1], fs_val), "CH1 原始", channels=1)
                    st.pyplot(fig)
                with c2:
                    fig = plot_data((filtered[:, 0:1], fs_val), "CH1 处理后", channels=1)
                    st.pyplot(fig)
                st.markdown("---")

            elif view == "ch1_freq":
                st.markdown("#### CH1 三角肌前束 — 频谱对比")
                c1, c2 = st.columns(2)
                with c1:
                    fig = plot_fft_spectrum(
                        (raw[:, 0:1], fs_val), channels=1,
                        title="CH1 原始频谱", freq_range=(0, 500), scale="dB",
                    )
                    st.pyplot(fig)
                with c2:
                    fig = plot_fft_spectrum(
                        (filtered[:, 0:1], fs_val), channels=1,
                        title="CH1 处理后频谱", freq_range=(0, 500), scale="dB",
                    )
                    st.pyplot(fig)
                st.markdown("---")

            elif view == "ch2_time":
                st.markdown("#### CH2 三角肌中束 — 时域对比")
                c1, c2 = st.columns(2)
                with c1:
                    fig = plot_data((raw[:, 1:2], fs_val), "CH2 原始", channels=1)
                    st.pyplot(fig)
                with c2:
                    fig = plot_data((filtered[:, 1:2], fs_val), "CH2 处理后", channels=1)
                    st.pyplot(fig)
                st.markdown("---")

            elif view == "ch2_freq":
                st.markdown("#### CH2 三角肌中束 — 频谱对比")
                c1, c2 = st.columns(2)
                with c1:
                    fig = plot_fft_spectrum(
                        (raw[:, 1:2], fs_val), channels=1,
                        title="CH2 原始频谱", freq_range=(0, 500), scale="dB",
                    )
                    st.pyplot(fig)
                with c2:
                    fig = plot_fft_spectrum(
                        (filtered[:, 1:2], fs_val), channels=1,
                        title="CH2 处理后频谱", freq_range=(0, 500), scale="dB",
                    )
                    st.pyplot(fig)
                st.markdown("---")

    # ============================================================
    # Tab 3: 活动检测
    # ============================================================
    with tab3:
        st.subheader("🎯 动作事件检测 (两阶段: 活动/静息分离 + 周期分割)")
        count = events["count"]

        # 指标卡片横排
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("检测周期数", count)
        if events["cycles"]:
            durs = [(e - s) / fs_val for s, e, _ in events["cycles"]]
            c2.metric("平均周期时长", f"{np.mean(durs):.2f} s")
            c3.metric("时长范围", f"{min(durs):.2f} – {max(durs):.2f} s")
        c4.metric("活动段数", len(events.get("active_segments", [])))

        # 静息基线信息
        bl = events.get("baseline", 0)
        th = events.get("threshold", 0)
        st.caption(
            f"静息基线: {bl:.4f} | 活动阈值: {th:.4f} | "
            f"标记: 🟢周期段(2) 🔵活动段(1) ⚪静息段(0)"
        )

        st.markdown("---")

        # 左侧: 周期表 + 活动段表 — 右侧: 包络图
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("#### 周期起止时间")
            if events["cycles"]:
                table_data = []
                for i, (s, e, pk) in enumerate(events["cycles"], start=1):
                    table_data.append({
                        "#": i,
                        "起始(s)": round(s / fs_val, 2),
                        "结束(s)": round(e / fs_val, 2),
                        "时长(s)": round((e - s) / fs_val, 2),
                        "波峰(s)": round(pk / fs_val, 2),
                    })
                st.dataframe(table_data, use_container_width=True, hide_index=True,
                             height=min(38 + 35 * len(table_data), 320))
            else:
                st.warning("未检测到有效周期")

            # 活动段/静息段摘要
            with st.expander("📋 活动段 & 静息段详情", expanded=False):
                if events.get("active_segments"):
                    st.markdown("**活动段:**")
                    adata = []
                    for i, (s, e) in enumerate(events["active_segments"], 1):
                        adata.append({
                            "#": i,
                            "起始(s)": round(s / fs_val, 2),
                            "结束(s)": round(e / fs_val, 2),
                            "时长(s)": round((e - s) / fs_val, 2),
                        })
                    st.dataframe(adata, use_container_width=True, hide_index=True)

                if events.get("rest_segments"):
                    st.markdown("**静息段:**")
                    rdata = []
                    for i, (s, e) in enumerate(events["rest_segments"], 1):
                        rdata.append({
                            "#": i,
                            "起始(s)": round(s / fs_val, 2),
                            "结束(s)": round(e / fs_val, 2),
                            "时长(s)": round((e - s) / fs_val, 2),
                        })
                    st.dataframe(rdata, use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("#### RMS 融合包络与周期边界")
            fig = plot_envelope_with_cycles(
                events["envelope"], fs_val, events["cycles"],
                envelope_ch1=events.get("envelope_ch1"),
                envelope_ch2=events.get("envelope_ch2"),
                threshold=th,
            )
            st.pyplot(fig)

        # 活动段标记图 (可折叠)
        with st.expander("📌 三级活动标记详情", expanded=False):
            st.caption("🟢 周期段(2) | 🔵 活动段(1) | ⚪ 静息段(0)")
            fig = plot_data(events["segment_data"], "活动标记", channels=3)
            st.pyplot(fig)

    # ============================================================
    # Tab 4: 特征展示 (自选)
    # ============================================================
    with tab4:
        st.subheader("📈 特征提取与展示")

        # 预处理时域图 (默认折叠)
        with st.expander("📌 预处理后双通道时域图 (上下文)", expanded=False):
            fig = plot_data((filtered, fs_val), "预处理后信号", channels=2)
            st.pyplot(fig)

        st.markdown("---")

        # 特征曲线选择
        st.markdown("#### 特征曲线")
        curve_opts = st.multiselect(
            "选择特征曲线",
            options=["rms", "mf", "mdf", "ratio"],
            default=["rms", "ratio"],
            format_func=lambda x: {
                "rms": "RMS 包络曲线",
                "mf": "中位频率 (MF) 曲线",
                "mdf": "平均功率频率 (MDF) 曲线",
                "ratio": "CH2/CH1 RMS 比值曲线",
            }[x],
            label_visibility="collapsed",
            key="tab4_curves",
        )
        if curve_opts:
            fig = viz_feature_curves(curves, fs_val, cycles=events["cycles"],
                                     selected_features=curve_opts)
            st.pyplot(fig)

        st.markdown("---")

        # 特征表格
        st.markdown("#### 周期级特征表")
        all_feature_cols = [
            c for c in features_df.columns
            if c not in ("cycle_id", "start_idx", "end_idx", "start_time",
                         "end_time", "duration", "action_label", "quality_label",
                         "abnormal_type", "label_source")
        ]
        selected_cols = st.multiselect(
            "选择特征列",
            options=all_feature_cols,
            default=[c for c in DEFAULT_SELECTED_FEATURES if c in all_feature_cols],
            label_visibility="collapsed",
            key="tab4_table",
        )

        meta_cols = ["cycle_id", "start_time", "end_time", "duration",
                     "action_label", "quality_label", "abnormal_type"]
        display_cols = [c for c in meta_cols if c in features_df.columns]
        display_cols += [c for c in selected_cols if c in features_df.columns]

        if not features_df.empty:
            st.dataframe(features_df[display_cols], use_container_width=True,
                         hide_index=True, height=min(38 + 35 * len(features_df), 400))
            st.caption(f"共 {len(features_df)} 个周期")

    # ============================================================
    # Tab 5: 分类与质量
    # ============================================================
    with tab5:
        st.subheader("🧠 动作分类与质量评估")
        st.caption("⚠️ 动作质量判断为辅助判断，不作为严格医学或运动学评价结果")

        # ---- 动作分类 ----
        st.markdown("### 📊 动作类型识别")
        overall_action = action_result.get("overall_action", "N/A")
        conf = action_result.get("overall_confidence", 0)
        votes = action_result.get("vote_counts", {})

        c1, c2 = st.columns(2)
        c1.metric("整体疑似动作", overall_action)
        vote_str = "  |  ".join(f"{k}: {v}票" for k, v in votes.items())
        c2.metric("投票分布", vote_str)

        if action_result.get("cycle_results"):
            st.markdown("**周期级分类结果：**")
            data = []
            for r in action_result["cycle_results"]:
                consistent = r.get("consistent", True)
                data.append({
                    "周期": r["cycle_id"],
                    "起始(s)": r["start_time"],
                    "时长(s)": r["duration"],
                    "预测": r["prediction"],
                    "置信度": f"{r.get('confidence', 1):.1%}",
                    "一致": "✓" if consistent else "⚠",
                })
            st.dataframe(data, use_container_width=True, hide_index=True)

            inconsistent = [r for r in action_result["cycle_results"]
                            if not r.get("consistent", True)]
            if inconsistent:
                st.warning(
                    f"⚠️ {len(inconsistent)} 个周期分类结果与整体不一致，"
                    f"可能与动作不标准、肌肉代偿或事件分割误差有关。"
                )

        st.markdown("---")

        # ---- 质量评估 ----
        st.markdown("### ⚙️ 动作质量辅助判断")
        c1, c2 = st.columns(2)
        c1.metric("✅ 标准周期", quality_result.get("standard_count", 0))
        c2.metric("⚠️ 不标准周期", quality_result.get("nonstandard_count", 0))

        if quality_result.get("cycle_results"):
            st.markdown("**周期级质量详情：**")
            qdata = []
            for r in quality_result["cycle_results"]:
                qdata.append({
                    "周期": r["cycle_id"],
                    "时长(s)": r["duration"],
                    "CH2/CH1": r["ratio"],
                    "质量": r["quality"],
                    "异常类型": r.get("abnormal_type") or "-",
                })
            st.dataframe(qdata, use_container_width=True, hide_index=True)

            abnormal = [r for r in quality_result["cycle_results"]
                        if r.get("explanation")]
            if abnormal:
                with st.expander("🔍 不标准周期详细解释", expanded=True):
                    for r in abnormal:
                        st.markdown(f"**周期 {r['cycle_id']}** "
                                    f"({r['start_time']}s – {r['end_time']}s, "
                                    f"CH2/CH1 = {r['ratio']})：")
                        st.info(r["explanation"])
