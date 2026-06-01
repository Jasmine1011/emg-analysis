# -*- coding: utf-8 -*-
"""
app.py — 双通道表面肌电信号处理与分析平台 v4.1

流程: 上传 .mat → 一键分析 → 5 个 Tab 展示结果
架构: FileResult dataclass + 展平错误处理
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import data_loader
from src.visualization_signal import plot_data, plot_fft_spectrum
from app.style import CUSTOM_CSS
from app.models import FileResult

from src.preprocessing import preprocess
from src.event_detection import detect_events
from src.features import compute_feature_curves, extract_all_cycles, load_labels, DEFAULT_SELECTED_FEATURES
from src.visualization import plot_feature_curves as viz_fc, plot_envelope_with_cycles
from src.prediction import predict_action, predict_quality

st.set_page_config(page_title="EMG 分析平台", page_icon="💪", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

for k, v in {"result": None, "analysis_done": False}.items():
    if k not in st.session_state: st.session_state[k] = v

# ================================================================
# 侧边栏
# ================================================================
with st.sidebar:
    st.markdown("### 📁 数据上传")
    uploaded = st.file_uploader("选择 .mat 文件", type=["mat"], label_visibility="collapsed")

    if uploaded:
        # 新文件 → 自动加载
        if st.session_state.result is None or st.session_state.result.fname != uploaded.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                tmp.write(uploaded.read()); tmp_path = tmp.name
            raw, fs = data_loader(tmp_path)
            os.unlink(tmp_path)
            st.session_state.result = FileResult(fname=uploaded.name, raw=raw, fs=fs)
            st.session_state.analysis_done = False
            st.session_state._params = {}  # 触发自动分析

        r = st.session_state.result
        st.success(f"**{r.fname}**")
        st.caption(f"采样率: {r.fs} Hz | 时长: {r.raw.shape[0]/r.fs:.0f}s")

    st.markdown("---")

    with st.expander("⚙️ 高级设置", expanded=False):
        p = {}
        p["notch_freq"] = st.number_input("工频频率 (Hz)", value=50.0, step=10.0)
        p["fc_high"] = st.number_input("高通截止 (Hz)", value=20.0, step=5.0)
        p["fc_low"] = st.number_input("低通截止 (Hz)", value=450.0, step=10.0)
        p["smooth_win"] = st.slider("平滑窗口 (s)", 0.05, 1.0, 0.300, 0.05)
        p["min_dur"] = st.slider("最小周期 (s)", 0.5, 3.0, 1.5, 0.1)
        p["max_dur"] = st.slider("最大周期 (s)", 4.0, 15.0, 8.0, 0.5)
        p["target_k"] = st.slider("预期周期数", 2, 8, 5)
        p["activity_sigma"] = st.slider("静息阈值倍数", 0.5, 15.0, 3.0, 0.5,
                                         help="越大越不敏感")

    st.markdown("---")
    # 首次上传 → 自动分析；调整参数后 → 手动重新分析
    if st.session_state.analysis_done:
        if st.button("🔄 重新分析", use_container_width=True, type="primary"):
            st.session_state._params = p
            st.session_state.analysis_done = False
            st.session_state.result.filtered = None
            st.rerun()
    elif not uploaded:
        pass  # 无文件，不显示按钮
    # else: 首次上传自动触发，不需要按钮

# ================================================================
# 分析管线
# ================================================================
def analyze(r, p):
    """展平错误处理 — 每步独立 catch"""
    # Step 1: 预处理
    try:
        r.filtered, _ = preprocess(r.raw, r.fs,
            apply_dc_removal=True, apply_amplitude_scaling=True, apply_notch=True,
            notch_freq=p["notch_freq"], fc_high=p["fc_high"], fc_low=p["fc_low"])
    except Exception as e:
        r.set_error(f"预处理失败: {e}")
        return

    # Step 2: 事件检测
    try:
        r.events = detect_events(r.filtered, r.fs,
            smooth_window=p["smooth_win"], min_duration=p["min_dur"],
            max_duration=p["max_dur"], target_k=p["target_k"],
            activity_sigma=p["activity_sigma"])
    except Exception as e:
        r.set_error(f"事件检测失败: {e}")
        return

    if r.cycle_count == 0:
        return

    # Step 3: 特征提取
    try:
        r.curves = compute_feature_curves(r.filtered, r.fs)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lb = load_labels(os.path.join(root, "data", "labels.csv"))
        r.features_df = extract_all_cycles(r.filtered, r.fs, r.events["cycles"],
                                           filename=r.fname, labels_df=lb)
    except Exception as e:
        r.set_error(f"特征提取失败: {e}")
        return

    # Step 4: 动作分类
    try:
        act = predict_action(r.filtered, r.fs, r.events["cycles"])
        r.action = act.get("overall_action", "?")
        r.action_votes = act.get("vote_counts", {})
        r.action_cycles = act.get("cycle_results", [])
    except Exception as e:
        r.action = f"错误"

    # Step 5: 质量评估
    try:
        qual = predict_quality(r.filtered, r.fs, r.events["cycles"],
                               action_label=r.action)
        r.std_count = qual.get("standard_count", 0)
        r.nonstd_count = qual.get("nonstandard_count", 0)
        r.quality_cycles = qual.get("cycle_results", [])
    except Exception as e:
        r.std_count, r.nonstd_count = 0, 0


if (st.session_state.result is not None and not st.session_state.analysis_done
        and st.session_state.get("_params") is not None):
    r = st.session_state.result
    with st.spinner("分析中..."):
        analyze(r, st.session_state._params)
    st.session_state.analysis_done = True
    if r.is_ok:
        st.success(f"🎉 分析完成 — {r.cycle_count} 周期, 动作: {r.action}")
    elif r.error:
        st.error(f"❌ {r.error}")
    else:
        st.warning(f"⚠️ 0 周期 — 尝试调低静息阈值倍数")

# ================================================================
# Tab 结构
# ================================================================
r = st.session_state.result
if r is None:
    st.title("💪 双通道表面肌电信号处理与分析")
    st.caption("CH1: 三角肌前束 | CH2: 三角肌中束 | 前平举·侧平举·推肩 | ⚠️ 质量判断为辅助参考")
    st.info("👈 请上传 .mat 文件，将自动开始分析")
elif not st.session_state.analysis_done:
    st.title(f"💪 {r.fname}")
    st.info("分析中...")
else:
    st.title(f"💪 {r.fname}")
    fs_val = r.fs

    tabs = st.tabs(["📊 原始信号", "🔧 预处理", "🎯 活动检测",
                     "📈 特征展示", "🧠 分类与质量"])

    # ---- Tab 1 ----
    with tabs[0]:
        st.subheader("📊 原始 EMG 信号")
        st.markdown(f"采样率: {fs_val} Hz | 时长: {r.raw.shape[0]/fs_val:.1f}s")
        st.pyplot(plot_data((r.raw, fs_val), "原始信号", channels=2))
        st.pyplot(plot_fft_spectrum((r.raw, fs_val), freq_range=(0, 500), scale="dB"))

    # ---- Tab 2 ----
    with tabs[1]:
        st.subheader("🔧 预处理结果")
        if r.filtered is None:
            st.warning("预处理未完成")
        else:
            views = st.multiselect("对比视图",
                options=["ch1_time","ch1_freq","ch2_time","ch2_freq"],
                default=["ch1_time","ch2_time"],
                format_func=lambda x: {"ch1_time":"CH1 时域","ch1_freq":"CH1 频谱",
                    "ch2_time":"CH2 时域","ch2_freq":"CH2 频谱"}[x],
                label_visibility="collapsed")
            for v in views:
                if v == "ch1_time":
                    c1,c2=st.columns(2)
                    c1.pyplot(plot_data((r.raw[:,0:1],fs_val),"CH1 原始",channels=1))
                    c2.pyplot(plot_data((r.filtered[:,0:1],fs_val),"CH1 处理后",channels=1))
                elif v == "ch1_freq":
                    c1,c2=st.columns(2)
                    c1.pyplot(plot_fft_spectrum((r.raw[:,0:1],fs_val),channels=1,
                              title="CH1 原始频谱",freq_range=(0,500),scale="dB"))
                    c2.pyplot(plot_fft_spectrum((r.filtered[:,0:1],fs_val),channels=1,
                              title="CH1 处理后频谱",freq_range=(0,500),scale="dB"))
                elif v == "ch2_time":
                    c1,c2=st.columns(2)
                    c1.pyplot(plot_data((r.raw[:,1:2],fs_val),"CH2 原始",channels=1))
                    c2.pyplot(plot_data((r.filtered[:,1:2],fs_val),"CH2 处理后",channels=1))
                elif v == "ch2_freq":
                    c1,c2=st.columns(2)
                    c1.pyplot(plot_fft_spectrum((r.raw[:,1:2],fs_val),channels=1,
                              title="CH2 原始频谱",freq_range=(0,500),scale="dB"))
                    c2.pyplot(plot_fft_spectrum((r.filtered[:,1:2],fs_val),channels=1,
                              title="CH2 处理后频谱",freq_range=(0,500),scale="dB"))
                st.markdown("---")

    # ---- Tab 3 ----
    with tabs[2]:
        st.subheader("🎯 动作事件检测")
        if r.events is None:
            st.warning("事件检测未完成")
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("周期数", r.cycle_count)
            if r.events["cycles"]:
                durs=[(e-s)/fs_val for s,e,_ in r.events["cycles"]]
                c2.metric("均长", f"{np.mean(durs):.2f}s")
                c3.metric("范围", f"{min(durs):.2f}–{max(durs):.2f}s")
            c4.metric("活动段", len(r.active_segments))
            st.caption(f"基线: {r.baseline:.4f} | 阈值: {r.threshold:.4f}")
            st.markdown("---")
            cl,cr=st.columns([1,2])
            with cl:
                st.markdown("#### 周期表")
                if r.events["cycles"]:
                    td=[{"#":i+1,"始(s)":round(s/fs_val,2),"止(s)":round(e/fs_val,2),
                         "长(s)":round((e-s)/fs_val,2),"峰(s)":round(pk/fs_val,2)}
                        for i,(s,e,pk) in enumerate(r.events["cycles"])]
                    st.dataframe(td, use_container_width=True, hide_index=True,
                                 height=min(38+35*len(td),320))
                else:
                    st.warning("无周期")
                with st.expander("📋 活动/静息段", expanded=False):
                    if r.active_segments:
                        st.caption("**活动段**")
                        st.dataframe([{"#":i+1,"始":round(s/fs_val,2),"止":round(e/fs_val,2),
                                      "长":round((e-s)/fs_val,2)}
                                      for i,(s,e) in enumerate(r.active_segments)],
                                     use_container_width=True, hide_index=True)
                    if r.rest_segments:
                        st.caption("**静息段**")
                        st.dataframe([{"#":i+1,"始":round(s/fs_val,2),"止":round(e/fs_val,2),
                                      "长":round((e-s)/fs_val,2)}
                                      for i,(s,e) in enumerate(r.rest_segments)],
                                     use_container_width=True, hide_index=True)
            with cr:
                st.markdown("#### 包络与周期边界")
                if r.envelope is not None:
                    st.pyplot(plot_envelope_with_cycles(r.envelope, fs_val,
                        r.events["cycles"], r.env_ch1, r.env_ch2, r.threshold))
            with st.expander("📌 活动标记", expanded=False):
                if r.segment_data is not None:
                    st.pyplot(plot_data(r.segment_data, "活动标记", channels=3))

    # ---- Tab 4 ----
    with tabs[3]:
        st.subheader("📈 特征展示")
        if r.filtered is not None:
            with st.expander("📌 预处理时域图", expanded=False):
                st.pyplot(plot_data((r.filtered, fs_val), "预处理后", channels=2))
        if r.curves is not None:
            st.markdown("#### 特征曲线")
            co=st.multiselect("选择曲线", ["rms","mf","mdf","ratio"],
                default=["rms","ratio"],
                format_func=lambda x: {"rms":"RMS","mf":"MF","mdf":"MDF","ratio":"CH2/CH1比值"}[x],
                label_visibility="collapsed", key="t4c")
            if co:
                st.pyplot(viz_fc(r.curves, fs_val,
                    cycles=r.events["cycles"] if r.events else None, selected_features=co))
        if r.features_df is not None:
            st.markdown("---")
            st.markdown("#### 周期特征表")
            fc_all=[c for c in r.features_df.columns
                    if c not in ("cycle_id","start_idx","end_idx","start_time","end_time",
                                 "duration","action_label","quality_label","abnormal_type","label_source")]
            sel=st.multiselect("选择列", fc_all,
                default=[c for c in DEFAULT_SELECTED_FEATURES if c in fc_all],
                label_visibility="collapsed", key="t4t")
            mc=["cycle_id","start_time","end_time","duration","action_label","quality_label","abnormal_type"]
            dc=[c for c in mc if c in r.features_df.columns]
            dc+=[c for c in sel if c in r.features_df.columns]
            st.dataframe(r.features_df[dc], use_container_width=True, hide_index=True,
                         height=min(38+35*len(r.features_df),400))
            st.caption(f"共 {len(r.features_df)} 周期")

    # ---- Tab 5 ----
    with tabs[4]:
        st.subheader("🧠 分类与质量")
        st.caption("⚠️ 质量判断为辅助参考")

        st.markdown("### 📊 动作类型")
        c1,c2=st.columns(2)
        c1.metric("整体动作", r.action or "?")
        c2.metric("投票", " | ".join(f"{k}:{v}" for k,v in r.action_votes.items()) if r.action_votes else "N/A")
        if r.action_cycles:
            st.markdown("**周期级:**")
            st.dataframe([{"周期":x["cycle_id"],"始(s)":x["start_time"],"长(s)":x["duration"],
                           "预测":x["prediction"],"置信":f"{x.get('confidence',1):.1%}",
                           "一致":"✓" if x.get("consistent",True) else "⚠"}
                          for x in r.action_cycles], use_container_width=True, hide_index=True)
            bad=[x for x in r.action_cycles if not x.get("consistent",True)]
            if bad: st.warning(f"⚠️ {len(bad)} 周期不一致")

        st.markdown("---")
        st.markdown("### ⚙️ 质量判断")
        c1,c2=st.columns(2)
        c1.metric("✅ 标准", r.std_count)
        c2.metric("⚠️ 不标准", r.nonstd_count)
        if r.quality_cycles:
            st.dataframe([{"周期":x["cycle_id"],"长(s)":x["duration"],"CH2/CH1":x["ratio"],
                           "质量":x["quality"],"异常":x.get("abnormal_type") or "-"}
                          for x in r.quality_cycles], use_container_width=True, hide_index=True)
            ab=[x for x in r.quality_cycles if x.get("explanation")]
            if ab:
                with st.expander("🔍 异常解释", expanded=True):
                    for x in ab:
                        st.info(f"周期{x['cycle_id']}: {x['explanation']}")
