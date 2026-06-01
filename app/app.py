# -*- coding: utf-8 -*-
"""
app.py — 双通道表面肌电信号处理与分析平台 v3.1

一键式流程: 上传多个 .mat → 开始分析 → 自动完成全管线 → 结果展示
支持多文件批量分析，侧边栏切换查看。
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
st.set_page_config(page_title="EMG 信号分析平台", page_icon="💪", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# 初始化
# ------------------------------------------------------------
for k, v in {"analysis_done": False, "results": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------
# 侧边栏
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 数据上传")
    uploaded_files = st.file_uploader("选择 .mat 文件（可多选）", type=["mat"],
                                       accept_multiple_files=True,
                                       label_visibility="collapsed")

    file_names = []
    if uploaded_files:
        for uf in uploaded_files:
            file_names.append(uf.name)
            if uf.name not in st.session_state.results:
                # 新文件：加载原始数据
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                raw, fs = data_loader(tmp_path)
                os.unlink(tmp_path)
                st.session_state.results[uf.name] = {"raw": raw, "fs": fs}

        st.success(f"已加载 {len(uploaded_files)} 个文件")
        if len(file_names) <= 5:
            for fn in file_names:
                r = st.session_state.results[fn]
                dur = r["raw"].shape[0] / r["fs"]
                st.caption(f"📄 {fn} ({dur:.0f}s)")

    st.markdown("---")

    with st.expander("⚙️ 高级设置", expanded=False):
        notch_freq = st.number_input("工频频率 (Hz)", value=50.0, step=10.0)
        fc_high = st.number_input("高通截止 (Hz)", value=20.0, step=5.0)
        fc_low = st.number_input("低通截止 (Hz)", value=450.0, step=10.0)
        smooth_win = st.slider("平滑窗口 (s)", 0.05, 1.0, 0.300, 0.05)
        min_dur = st.slider("最小周期 (s)", 0.5, 3.0, 1.5, 0.1)
        max_dur = st.slider("最大周期 (s)", 4.0, 15.0, 8.0, 0.5)
        target_k = st.slider("预期周期数", 2, 8, 5)
        st.caption("活动/静息检测")
        activity_sigma = st.slider("静息阈值倍数", 0.5, 15.0, 3.0, 0.5,
                                   help="越大越不敏感")

    st.markdown("---")
    btn_disabled = not uploaded_files
    if st.button("🚀 开始分析", use_container_width=True, type="primary",
                 disabled=btn_disabled):
        st.session_state._params = dict(
            notch_freq=notch_freq, fc_high=fc_high, fc_low=fc_low,
            smooth_win=smooth_win, min_dur=min_dur, max_dur=max_dur,
            target_k=target_k, activity_sigma=activity_sigma,
        )
        st.session_state.analysis_done = False
        st.session_state.current_file = None
        st.rerun()

    # 分析完成后显示文件选择器 + 导出按钮
    if st.session_state.analysis_done and st.session_state.results:
        analyzed = [fn for fn, r in st.session_state.results.items()
                    if r.get("filtered") is not None]
        if len(analyzed) >= 1:
            st.markdown("---")
            st.markdown("### 📂 切换文件")
            idx = 0
            if st.session_state.get("current_file") in analyzed:
                idx = analyzed.index(st.session_state["current_file"])
            selected = st.selectbox("选择查看的文件", analyzed, index=idx,
                                    key="file_selector",
                                    label_visibility="collapsed")
            if selected != st.session_state.get("current_file"):
                st.session_state.current_file = selected
                st.rerun()

        # 导出预测 CSV
        if analyzed:
            st.markdown("---")
            ACTION_TO_LABEL = {"前平举": 1, "侧平举": 2, "推肩": 3}
            rows = []
            for fn in analyzed:
                fid = fn.replace(".mat", "")
                r = st.session_state.results[fn]
                act = r.get("action_result", {}).get("overall_action", "")
                label = ACTION_TO_LABEL.get(act, "")
                rows.append(f"{fid},{label}")
            csv_content = "ID,Pred_Label\n" + "\n".join(rows)
            st.download_button("📥 导出预测 CSV", data=csv_content,
                               file_name="Pred_Labels_Group03_Submit1.csv",
                               mime="text/csv", use_container_width=True)

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
# 一键分析
# ------------------------------------------------------------
def analyze_one_file(fname, p, progress_bar, step_idx, total_files):
    """分析单个文件，结果写回 session_state.results[fname]"""
    r = st.session_state.results[fname]
    try:
        _analyze_one_file(fname, p, r, progress_bar, step_idx, total_files)
    except Exception as exc:
        r["error"] = str(exc)
        r["filtered"] = np.zeros((100, 2))
        r["events"] = {"count": 0, "cycles": [], "active_segments": [],
                       "rest_segments": [], "envelope": np.zeros(100),
                       "envelope_ch1": np.zeros(100), "envelope_ch2": np.zeros(100),
                       "segment_data": (np.zeros((100,3)), 2000),
                       "baseline": 0, "threshold": 0}
        r["curves"] = {}
        r["features_df"] = None
        r["action_result"] = {"overall_action": f"错误: {exc}", "cycle_results": [],
                               "vote_counts": {}}
        r["quality_result"] = {"standard_count": 0, "nonstandard_count": 0,
                                "overall_summary": f"错误: {exc}", "cycle_results": []}
        progress_bar.progress((step_idx + 1) / total_files,
                              text=f"❌ {fname} (失败)")


def _analyze_one_file(fname, p, r, progress_bar, step_idx, total_files):
    raw = r["raw"]
    fs_val = r["fs"]

    # Step 1: 预处理
    filtered, _ = preprocess(raw, fs_val,
        apply_dc_removal=True, apply_amplitude_scaling=True, apply_notch=True,
        notch_freq=p["notch_freq"], fc_high=p["fc_high"], fc_low=p["fc_low"])
    r["filtered"] = filtered

    # Step 2: 事件检测
    events = detect_events(filtered, fs_val,
        smooth_window=p["smooth_win"], min_duration=p["min_dur"],
        max_duration=p["max_dur"], target_k=p["target_k"],
        activity_sigma=p["activity_sigma"])
    r["events"] = events

    # 无有效周期 → 跳过后续步骤
    if events["count"] == 0:
        r["curves"] = {}
        r["features_df"] = None
        r["action_result"] = {"overall_action": "无周期", "cycle_results": [],
                               "vote_counts": {}}
        r["quality_result"] = {"standard_count": 0, "nonstandard_count": 0,
                                "overall_summary": "无周期", "cycle_results": []}
        progress_bar.progress((step_idx + 1) / total_files,
                              text=f"⚠️ {fname} (0 周期, 跳过)")
        return

    # Step 3: 特征提取
    try:
        curves = compute_feature_curves(filtered, fs_val)
        r["curves"] = curves

        labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "data", "labels.csv")
        labels_df = load_labels(labels_path)
        features_df = extract_all_cycles(filtered, fs_val, events["cycles"],
                                         filename=fname, labels_df=labels_df)
        r["features_df"] = features_df

        # Step 4: 动作分类
        action_result = predict_action(filtered, fs_val, events["cycles"])
        r["action_result"] = action_result

        # Step 5: 质量评估
        quality_result = predict_quality(filtered, fs_val, events["cycles"],
                                         action_label=action_result.get("overall_action", ""))
        r["quality_result"] = quality_result

        progress_bar.progress((step_idx + 1) / total_files,
                              text=f"✅ {fname} ({events['count']} 周期, {action_result.get('overall_action','?')})")
    except Exception as e:
        r["curves"] = {}
        r["features_df"] = None
        r["action_result"] = {"overall_action": f"错误: {e}", "cycle_results": [],
                               "vote_counts": {}}
        r["quality_result"] = {"standard_count": 0, "nonstandard_count": 0,
                                "overall_summary": f"错误: {e}", "cycle_results": []}
        progress_bar.progress((step_idx + 1) / total_files,
                              text=f"❌ {fname} (失败: {str(e)[:40]})")


def run_analysis():
    """遍历所有上传文件执行分析"""
    files_to_analyze = [fn for fn, r in st.session_state.results.items()
                        if r.get("filtered") is None]
    p = st.session_state._params
    total = len(files_to_analyze)

    progress_bar = st.progress(0, text=f"分析中: 0/{total} 文件...")
    status_container = st.empty()

    try:
        for i, fname in enumerate(files_to_analyze):
            progress_bar.progress(i / total, text=f"🔍 {fname} ({i+1}/{total})...")
            analyze_one_file(fname, p, progress_bar, i, total)

        progress_bar.progress(1.0, text="✅ 分析完成！")
        st.session_state.analysis_done = True
        st.session_state.current_file = files_to_analyze[0]

        # 汇总
        summary_parts = []
        for fn in files_to_analyze:
            r = st.session_state.results[fn]
            if r.get("error"):
                summary_parts.append(f"**{fn}**: ❌ {r['error'][:30]}")
            else:
                n = r["events"]["count"]
                a = r["action_result"].get("overall_action", "?")
                summary_parts.append(f"**{fn}**: {n}周期, {a}")
        status_container.success("🎉 分析完成 | " + " | ".join(summary_parts))

    except Exception as exc:
        progress_bar.progress(1.0, text="❌ 分析失败")
        status_container.error(f"分析出错: {exc}")
        st.session_state.analysis_done = False


# ---- 触发分析 ----
if (st.session_state.results and not st.session_state.analysis_done
        and '_params' in st.session_state):
    run_analysis()

# ------------------------------------------------------------
# 获取当前文件数据
# ------------------------------------------------------------
def get_current_data():
    """返回当前选中文件的分析数据"""
    current = st.session_state.get("current_file")
    if not current:
        return None
    r = st.session_state.results.get(current, {})
    if r.get("filtered") is None:
        return None
    return {
        "fname": current,
        "raw": r["raw"],
        "fs": r["fs"],
        "filtered": r["filtered"],
        "events": r["events"],
        "curves": r["curves"],
        "features_df": r["features_df"],
        "action_result": r["action_result"],
        "quality_result": r["quality_result"],
    }

# ------------------------------------------------------------
# Tab 结构
# ------------------------------------------------------------
if not st.session_state.analysis_done:
    if not st.session_state.results:
        st.info("👈 请上传 .mat 文件并点击「开始分析」")
    else:
        st.info("👈 请点击「开始分析」按钮")
else:
    d = get_current_data()
    if d is None:
        st.info("👈 请在侧边栏选择一个已分析的文件")
    elif d["events"]["count"] == 0:
        err = st.session_state.results.get(st.session_state.current_file, {}).get("error", "未知错误")
        st.error(f"❌ 文件分析失败: {err}")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 原始信号", "🔧 预处理结果", "🎯 活动检测",
             "📈 特征展示", "🧠 分类与质量"]
        )

        raw = d["raw"]
        fs_val = d["fs"]
        filtered = d["filtered"]
        events = d["events"]
        curves = d["curves"]
        features_df = d["features_df"]
        action_result = d["action_result"]
        quality_result = d["quality_result"]

        # ====== Tab 1: 原始信号 ======
        with tab1:
            st.subheader(f"📊 原始 EMG 信号 — {d['fname']}")
            st.markdown(f"采样率: {fs_val} Hz | 时长: {raw.shape[0]/fs_val:.1f}s")

            st.markdown("#### 双通道时域波形")
            fig = plot_data((raw, fs_val), "原始信号", channels=2)
            st.pyplot(fig)

            st.markdown("#### 幅度谱 (0–500 Hz, dB)")
            fig = plot_fft_spectrum((raw, fs_val), freq_range=(0, 500), scale="dB")
            st.pyplot(fig)

        # ====== Tab 2: 预处理结果 ======
        with tab2:
            st.subheader("🔧 预处理结果")
            st.caption("选择要查看的对比视图：")

            view_opts = st.multiselect(
                "对比视图",
                options=["ch1_time", "ch1_freq", "ch2_time", "ch2_freq"],
                default=["ch1_time", "ch2_time"],
                format_func=lambda x: {
                    "ch1_time": "CH1 时域对比", "ch1_freq": "CH1 频谱对比",
                    "ch2_time": "CH2 时域对比", "ch2_freq": "CH2 频谱对比",
                }[x],
                label_visibility="collapsed",
            )

            for view in view_opts:
                if view == "ch1_time":
                    st.markdown("#### CH1 时域对比")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = plot_data((raw[:, 0:1], fs_val), "CH1 原始", channels=1)
                        st.pyplot(fig)
                    with c2:
                        fig = plot_data((filtered[:, 0:1], fs_val), "CH1 处理后", channels=1)
                        st.pyplot(fig)
                elif view == "ch1_freq":
                    st.markdown("#### CH1 频谱对比")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = plot_fft_spectrum((raw[:, 0:1], fs_val), channels=1,
                                                title="CH1 原始频谱", freq_range=(0, 500), scale="dB")
                        st.pyplot(fig)
                    with c2:
                        fig = plot_fft_spectrum((filtered[:, 0:1], fs_val), channels=1,
                                                title="CH1 处理后频谱", freq_range=(0, 500), scale="dB")
                        st.pyplot(fig)
                elif view == "ch2_time":
                    st.markdown("#### CH2 时域对比")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = plot_data((raw[:, 1:2], fs_val), "CH2 原始", channels=1)
                        st.pyplot(fig)
                    with c2:
                        fig = plot_data((filtered[:, 1:2], fs_val), "CH2 处理后", channels=1)
                        st.pyplot(fig)
                elif view == "ch2_freq":
                    st.markdown("#### CH2 频谱对比")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = plot_fft_spectrum((raw[:, 1:2], fs_val), channels=1,
                                                title="CH2 原始频谱", freq_range=(0, 500), scale="dB")
                        st.pyplot(fig)
                    with c2:
                        fig = plot_fft_spectrum((filtered[:, 1:2], fs_val), channels=1,
                                                title="CH2 处理后频谱", freq_range=(0, 500), scale="dB")
                        st.pyplot(fig)
                st.markdown("---")

        # ====== Tab 3: 活动检测 ======
        with tab3:
            st.subheader("🎯 动作事件检测")
            count = events["count"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("检测周期数", count)
            if events["cycles"]:
                durs = [(e - s) / fs_val for s, e, _ in events["cycles"]]
                c2.metric("平均周期时长", f"{np.mean(durs):.2f} s")
                c3.metric("时长范围", f"{min(durs):.2f} – {max(durs):.2f} s")
            c4.metric("活动段数", len(events.get("active_segments", [])))

            bl = events.get("baseline", 0)
            th = events.get("threshold", 0)
            st.caption(f"静息基线: {bl:.4f} | 活动阈值: {th:.4f} | "
                       f"标记: 🟢周期段(2) 🔵活动段(1) ⚪静息段(0)")

            st.markdown("---")
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown("#### 周期起止时间")
                if events["cycles"]:
                    table_data = []
                    for i, (s, e, pk) in enumerate(events["cycles"], start=1):
                        table_data.append({
                            "#": i, "起始(s)": round(s / fs_val, 2),
                            "结束(s)": round(e / fs_val, 2),
                            "时长(s)": round((e - s) / fs_val, 2),
                            "波峰(s)": round(pk / fs_val, 2),
                        })
                    st.dataframe(table_data, use_container_width=True, hide_index=True,
                                 height=min(38 + 35 * len(table_data), 320))
                else:
                    st.warning("未检测到有效周期")

                with st.expander("📋 活动段 & 静息段详情", expanded=False):
                    if events.get("active_segments"):
                        st.markdown("**活动段:**")
                        adata = [{"#": i+1, "起始(s)": round(s/fs_val,2),
                                  "结束(s)": round(e/fs_val,2),
                                  "时长(s)": round((e-s)/fs_val,2)}
                                 for i, (s, e) in enumerate(events["active_segments"])]
                        st.dataframe(adata, use_container_width=True, hide_index=True)
                    if events.get("rest_segments"):
                        st.markdown("**静息段:**")
                        rdata = [{"#": i+1, "起始(s)": round(s/fs_val,2),
                                  "结束(s)": round(e/fs_val,2),
                                  "时长(s)": round((e-s)/fs_val,2)}
                                 for i, (s, e) in enumerate(events["rest_segments"])]
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

            with st.expander("📌 三级活动标记详情", expanded=False):
                st.caption("🟢 周期段(2) | 🔵 活动段(1) | ⚪ 静息段(0)")
                fig = plot_data(events["segment_data"], "活动标记", channels=3)
                st.pyplot(fig)

        # ====== Tab 4: 特征展示 ======
        with tab4:
            st.subheader("📈 特征提取与展示")

            with st.expander("📌 预处理后双通道时域图", expanded=False):
                fig = plot_data((filtered, fs_val), "预处理后信号", channels=2)
                st.pyplot(fig)

            st.markdown("---")
            st.markdown("#### 特征曲线")
            curve_opts = st.multiselect(
                "选择特征曲线",
                options=["rms", "mf", "mdf", "ratio"],
                default=["rms", "ratio"],
                format_func=lambda x: {
                    "rms": "RMS 包络曲线", "mf": "中位频率 (MF) 曲线",
                    "mdf": "平均功率频率 (MDF) 曲线", "ratio": "CH2/CH1 RMS 比值曲线",
                }[x],
                label_visibility="collapsed", key="tab4_curves",
            )
            if curve_opts:
                fig = viz_feature_curves(curves, fs_val, cycles=events["cycles"],
                                         selected_features=curve_opts)
                st.pyplot(fig)

            st.markdown("---")
            st.markdown("#### 周期级特征表")
            all_fc = [c for c in features_df.columns
                      if c not in ("cycle_id","start_idx","end_idx","start_time",
                                   "end_time","duration","action_label","quality_label",
                                   "abnormal_type","label_source")]
            selected_cols = st.multiselect(
                "选择特征列", options=all_fc,
                default=[c for c in DEFAULT_SELECTED_FEATURES if c in all_fc],
                label_visibility="collapsed", key="tab4_table",
            )
            meta_cols = ["cycle_id","start_time","end_time","duration",
                         "action_label","quality_label","abnormal_type"]
            display_cols = [c for c in meta_cols if c in features_df.columns]
            display_cols += [c for c in selected_cols if c in features_df.columns]
            if features_df is not None and not features_df.empty:
                st.dataframe(features_df[display_cols], use_container_width=True,
                             hide_index=True, height=min(38+35*len(features_df),400))
                st.caption(f"共 {len(features_df)} 个周期")

        # ====== Tab 5: 分类与质量 ======
        with tab5:
            st.subheader("🧠 动作分类与质量评估")
            st.caption("⚠️ 质量判断为辅助参考，不作为严格医学或运动学评价结果")

            st.markdown("### 📊 动作类型识别")
            oa = action_result.get("overall_action", "N/A")
            votes = action_result.get("vote_counts", {})

            c1, c2 = st.columns(2)
            c1.metric("整体疑似动作", oa)
            c2.metric("投票分布", " | ".join(f"{k}: {v}票" for k, v in votes.items()))

            if action_result.get("cycle_results"):
                st.markdown("**周期级分类结果：**")
                data = [{"周期": r["cycle_id"], "起始(s)": r["start_time"],
                         "时长(s)": r["duration"], "预测": r["prediction"],
                         "置信度": f"{r.get('confidence',1):.1%}",
                         "一致": "✓" if r.get("consistent", True) else "⚠"}
                        for r in action_result["cycle_results"]]
                st.dataframe(data, use_container_width=True, hide_index=True)
                inconsistent = [r for r in action_result["cycle_results"]
                                if not r.get("consistent", True)]
                if inconsistent:
                    st.warning(f"⚠️ {len(inconsistent)} 个周期与整体不一致")

            st.markdown("---")
            st.markdown("### ⚙️ 动作质量辅助判断")
            c1, c2 = st.columns(2)
            c1.metric("✅ 标准周期", quality_result.get("standard_count", 0))
            c2.metric("⚠️ 不标准周期", quality_result.get("nonstandard_count", 0))

            if quality_result.get("cycle_results"):
                st.markdown("**周期级质量详情：**")
                qdata = [{"周期": r["cycle_id"], "时长(s)": r["duration"],
                          "CH2/CH1": r["ratio"], "质量": r["quality"],
                          "异常类型": r.get("abnormal_type") or "-"}
                         for r in quality_result["cycle_results"]]
                st.dataframe(qdata, use_container_width=True, hide_index=True)
                abnormal = [r for r in quality_result["cycle_results"]
                            if r.get("explanation")]
                if abnormal:
                    with st.expander("🔍 不标准周期详细解释", expanded=True):
                        for r in abnormal:
                            st.markdown(f"**周期 {r['cycle_id']}** "
                                        f"({r['start_time']}s–{r['end_time']}s, "
                                        f"CH2/CH1={r['ratio']})：")
                            st.info(r["explanation"])
