# backup/ — 历史版本备份

本目录存放项目中已被替换/废弃的旧版模块文件。

---

## 第一版（TKEO 时代）

基于 TKEO (Teager-Kaiser Energy Operator) + 双阈值状态机的事件检测管线。
在项目 v1.0～v2.0 期间使用，后被 src/ 模块替代。

| 备份文件 | 说明 | 被替代原因 |
|---|---|---|
| `preprocessing_v1_tkeo.py` | 原始预处理模块。管线：幅值换算(×0.286/6) → 50Hz+谐波级联陷波 → 20Hz 高通 → 450Hz 低通。无去直流步骤。 | 缺少去直流、参数不可配置、返回 dict 格式不一致 |
| `event_detection_v1_tkeo.py` | TKEO + 双阈值状态机事件检测。依赖 preprocessing_v1_tkeo.py。功能包括：TKEO 能量算子、RMS 包络、双阈值状态机 (OnSigma=15, OffSigma=5)、自适应分段合并、双通道协同验证。 | 对噪声敏感、硬编码参数过多、与 labels.csv 不一致、不区分活动/静息段 |

### 被替代时间线

| 版本 | 预处理 | 事件检测 | 状态 |
|---|---|---|---|
| v1.0 | `emg/preprocessing.py` (TKEO) | `emg/event_detection.py` (TKEO 状态机) | 🔴 已删除 |
| v2.0 | `src/preprocessing.py` (统一可配置) | `src/event_detection.py` (Top-K 寻峰) | 🟡 已优化 |
| v2.1 | `src/preprocessing.py` (同上) | `src/event_detection.py` (两阶段: 活动/静息分离 + 阈值交叉法) | 🟢 当前 |

### 恢复方法

如需回退到 TKEO 版本（不推荐），将备份文件复制回 `emg/` 目录并去掉版本后缀：

```bash
cp backup/preprocessing_v1_tkeo.py emg/preprocessing.py
cp backup/event_detection_v1_tkeo.py emg/event_detection.py
```

然后修改 `emg/app.py` 中的导入语句，从 `src.preprocessing` 改回 `emg.preprocessing`。

---

*最后更新: 2026-05-31*
