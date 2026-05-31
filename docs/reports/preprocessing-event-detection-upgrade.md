# 预处理与事件检测模块升级说明

**日期**: 2026-05-31

---

## 一、为什么要升级

原有预处理和事件检测模块存在以下问题：

1. **两套不一致的实现**: `emg/preprocessing.py` (TKEO时代) 和 `dataset/quality_rules.py` (labels.csv时代) 的预处理管线不同：
   - 旧版 `emg/preprocessing.py`: 无去直流、高通+低通分离滤波、陷波级联卷积
   - `quality_rules.py`: 有去直流、20-450Hz bandpass单次滤波、逐谐波filtfilt陷波

2. **事件检测方法落后**: TKEO+双阈值状态机虽然能工作，但：
   - 对噪声敏感（TKEO放大高频噪声）
   - 需要大量硬编码阈值参数（OnSigma=15, OffSigma=5等）
   - 与 `labels.csv` 的生成方法不一致（labels.csv用Top-K寻峰法）

3. **参数不可配置**: 旧版所有滤波参数硬编码，用户无法在网页中调节

## 二、升级内容

### 2.1 预处理管线统一

| 步骤 | 旧版 (emg/) | 新版 (src/) | 依据 |
|---|---|---|---|
| 幅值换算 | ✅ ×0.286/6 | ✅ 可选 | illustration 第三节 |
| 去直流 | ❌ 缺失 | ✅ 可选 (默认开) | illustration 第三节 + quality_rules.py |
| 高通 | 20Hz 4阶 butter | 20Hz 4阶 butter ✅ 一致 | illustration 第三节 |
| 低通 | 450Hz 10阶 butter | 450Hz 10阶 butter ✅ 一致 | 用户指定 |
| 陷波 | 50Hz+谐波 级联卷积 | 50Hz+谐波 级联卷积 ✅ 一致 | 用户指定 |
| 滤波方式 | filtfilt 零相位 | filtfilt 零相位 ✅ 一致 | illustration 第三节 |

**关键改进**: 新增去直流步骤。实验表明去除基线漂移对于后续RMS包络计算和特征提取至关重要。

### 2.2 事件检测方法切换

| 特性 | TKEO + 双阈值状态机 | Top‑K 寻峰法 (新版) |
|---|---|---|
| 输入 | TKEO能量算子 | RMS包络 |
| 寻峰方法 | 状态机On/Off阈值 | prominence峰值突出度 |
| 边界确定 | 阈值交叉点 | 相邻峰间谷底 |
| 参数数量 | ~12个硬编码 | 7个可配置 |
| 自适应能力 | 固定阈值 | prominence自适应 |
| 与labels.csv一致 | ❌ 不一致 | ✅ 一致 |

**Top‑K 寻峰法优势**:
- Prominence 突出度衡量更鲁棒，不受基线漂移影响
- 谷底边界更符合生理学周期定义（一次完整动作的力学边界）
- 自适应 K 值可处理 4-6 次不等的重复次数

### 2.3 参数化改造

所有关键参数均可通过 Streamlit 控件调节：

**预处理参数** (侧边栏):
- 去直流开关
- 幅值换算开关
- 工频陷波开关 + 基频
- 高通/低通截止频率

**检测参数** (侧边栏):
- 平滑窗口 (0.05–1.0s)
- 最小/最大周期时长 (0.5–3.0s / 4.0–15.0s)
- 预期周期数 (2–8)

## 三、数据格式变化

| 模块 | 旧格式 | 新格式 |
|---|---|---|
| preprocess 输出 | `{'data': array, 'fs': fs}` | `(array, fs)` tuple |
| detect_events 输出 | `(segment_data, count, borders)` | `dict{cycles, borders, segment_data, count, envelope, ...}` |

新格式向后兼容 `emg/visualization_signal.py` 的 `plot_data()` 和 `plot_fft_spectrum()` 函数（它们接受 tuple）。

## 四、备份说明

- `emg/preprocessing.py` → `emg/preprocessing.py.bak`
- `emg/event_detection.py` → `emg/event_detection.py.bak`

如需回退到 TKEO 版本，恢复 .bak 文件并还原 `app.py` 中对应的 import 即可。
