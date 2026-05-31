# -*- coding: utf-8 -*-
"""
emg/style.py — Streamlit 自定义 CSS 样式

轻量视觉定制：主色调、卡片圆角、表格边框、侧边栏背景
"""

CUSTOM_CSS = """
<style>
    /* ===== 全局 ===== */
    .main .block-container {
        padding-top: 1.5rem;
    }

    /* 标题 */
    .main h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .main h2 {
        font-size: 1.3rem;
        font-weight: 600;
        color: #16213e;
    }
    .main h3 {
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* ===== 指标卡片 ===== */
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetric"] label {
        color: #6c757d;
        font-size: 0.8rem;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1f77b4;
        font-weight: 700;
        font-size: 1.6rem;
    }

    /* ===== 按钮 ===== */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f77b4, #2196F3);
        border: none;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 12px rgba(31,119,180,0.35);
        transform: translateY(-1px);
    }

    /* ===== 表格 ===== */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e9ecef;
    }
    [data-testid="stDataFrame"] thead th {
        background: #f1f3f5;
        font-weight: 600;
        font-size: 0.82rem;
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: #fafbfc;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }

    /* ===== 展开器 ===== */
    [data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: none;
    }

    /* ===== 信息/成功/警告框 ===== */
    .stAlert {
        border-radius: 10px;
    }

    /* ===== 进度条 ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1f77b4, #4CAF50);
    }

    /* ===== 分隔线 ===== */
    hr {
        margin: 1.5rem 0;
        border-color: #e9ecef;
    }

    /* ===== 多选/下拉 ===== */
    .stMultiSelect [data-baseweb="tag"] {
        border-radius: 6px;
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
</style>
"""
