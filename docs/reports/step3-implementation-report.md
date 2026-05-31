# 第3步实施报告：特征提取模块 + Streamlit 可视化

**日期**: 2026-05-31  
**状态**: ✅ 完成

---

## 一、变更摘要

将项目的事件检测方法从 TKEO+双阈值状态机统一切换为 Top‑K 寻峰法，实现了全部 17 个周期级 EMG 特征的提取，并完成了 Streamlit Tab 2‑4 的完整可视化功能。

## 二、文件变更清单

| 操作 | 文件 | 说明 |
|---|---|---|
| 📄 备份 | `emg/preprocessing.py.bak` | 保留 TKEO 时代原版 |
| 📄 备份 | `emg/event_detection.py.bak` | 保留 TKEO 时代原版 |
| ✨ 新建 | `src/__init__.py` | 包初始化文件 |
| ✨ 新建 | `src/preprocessing.py` | 统一可配置预处理模块 (199 行) |
| ✨ 新建 | `src/event_detection.py` | 优化版 Top‑K 寻峰法事件检测 (252 行) |
| ✨ 新建 | `src/features.py` | 全部 17 个特征提取 (330 行) |
| ✨ 新建 | `src/visualization.py` | 特征曲线和包络可视化 (224 行) |
| ✏️ 重写 | `emg/app.py` | Streamlit 完整应用 (340 行) |

## 三、各模块关键设计

### 3.1 预处理 (`src/preprocessing.py`)

**管线**: 幅值换算 → 去直流 → 高通 20Hz/4阶 → 低通 450Hz/10阶 → 50Hz+谐波陷波

**参数全部可配置**: 所有滤波步骤、频率、阶数均通过关键字参数控制。

**辅助函数**:
- `compute_rms_envelope()`: RMS 滑动窗口包络计算(支持重叠)
- `moving_average()`: 移动平均平滑

**数据格式**: 输入 `(ndarray(N,2), float fs)`, 输出 `(ndarray(N,2), float fs)`

### 3.2 事件检测 (`src/event_detection.py`)

**算法**: Top‑K prominence 寻峰 + 谷底边界 + 物理时长过滤

**流程**:
1. 双通道 RMS 包络 (200ms 窗, 50% 重叠)
2. 归一化 + 融合 `envelope = 0.5*env1 + 0.5*env2`
3. 平滑包络 (可配窗口)
4. `scipy.signal.find_peaks` + `peak_prominences`
5. 自适应 K 值 (`_choose_k`)
6. 谷底边界 (`_boundaries`)
7. 时长过滤 [1.5s, 8.0s]

**输出**: dict 含 cycles, borders, segment_data, count, envelope

**核心优化** (相比 `dataset/quality_rules.py`):
- 使用 `compute_rms_envelope` 替代 Hilbert 包络 (更稳健)
- 新增 `prominence_factor` 参数控制最小突出度
- 新增 `segment_data` 生成 (兼容 app.py 画图)

### 3.3 特征提取 (`src/features.py`)

**特征曲线** (滑动窗口):
- RMS 曲线: 200ms 窗, 100ms 步长
- MF/MDF 曲线: 500ms 窗, 100ms 步长
- CH2/CH1 RMS 比值曲线

**周期级特征** (17 个全部实现):

| 类别 | 特征 | 每通道2个 | 跨通道 |
|---|---|---|---|
| 时域 | RMS, MAV, VAR, WL, ZC, SSC, IEMG | 14 个 | - |
| 频域 | MF, MDF, PF | 6 个 | - |
| 协同 | ratio, diff, ratio_ch1, ratio_ch2, corr, peak_t1, peak_t2, act_diff | - | 8 个 |
| **合计** | | **20** | **8** |

**标签匹配**: 以新检测边界为准, 与 `labels.csv` 按 200ms 容差双向匹配。未匹配的标记为 "未标注"。

### 3.4 可视化 (`src/visualization.py`)

- `plot_envelope_with_cycles()`: 包络图 + 周期边界 + 波峰标记 + 通道叠加
- `plot_feature_curves()`: 可选子图 (RMS/MF/MDF/ratio), 自动标注周期边界

### 3.5 Streamlit 应用 (`emg/app.py`)

**Tab 结构**:
- Tab 1 (原始信号): 时域 + 频谱图 (保持原有)
- Tab 2 (预处理结果): 新增参数控件, CH1/CH2 处理前后 4 图对比
- Tab 3 (活动检测): 周期表 + 活动标记图 + 包络图 + 检测参数控件
- Tab 4 (特征展示): 特征曲线(可选) + 特征表格(可勾选列) + 标签匹配
- Tab 5 (分类与质量): 占位 (第4步实现)

## 四、测试结果

### 4.1 单元测试
- ✅ 预处理管线: 参数化过滤正常
- ✅ 事件检测: 合成周期信号检测正确
- ✅ 特征提取: 全部 28 个键值输出正确

### 4.2 真实数据测试

| 文件 | 预期周期 | 检测周期 | 标签匹配率 |
|---|---|---|---|
| emg_01_qpj_1.mat | 5 | 5 ✅ | 0/5 (边界偏差) |
| emg_02_qpj_1.mat | 4 | 4 ✅ | 2/4 |
| emg_05_cpj_1.mat | 5 | 5 ✅ | 5/5 ✅ |

**标签匹配差异原因**: 新检测使用 RMS 包络 + 谷底边界, 与 `quality_rules.py` 的 Hilbert 包络 + 谷底边界存在细微差异 (约 100-800 samples), 超过 200ms 容差。这符合设计预期 — 方案A: "以新检测结果为准, 不匹配标记为未标注"。第4步训练时将使用新检测边界重新生成标签。

### 4.3 可视化测试
- ✅ 包络标注图: 正常生成
- ✅ 特征曲线图: 正常生成 (含周期边界标注)

## 五、已知问题与后续建议

1. **标签匹配率**: 由于检测算法变更, 部分文件匹配率不高。建议第4步训练时直接用新检测算法重新生成 `labels.csv`。
2. **频域分辨率**: 500ms 窗 → 2Hz 分辨率, MF/MDF 曲线较粗糙。如需更精细曲线可增大窗口。
3. **Python 版本**: venv (`emg/.venv`) 链接已失效, 实际使用 Anaconda (`D:/Anaconda_202410/python.exe`)。建议重建 venv。
4. **性能**: 滑动窗口特征曲线对 20 秒信号计算约 200 次 FFT, 实时响应可接受 (~0.5s)。
