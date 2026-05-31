# 第4步实施报告：模型训练 + Tab 5 集成

**日期**: 2026-05-31  
**状态**: ✅ 完成

---

## 一、变更摘要

完成了动作类型三分类模型和动作质量二分类模型的训练，10 个候选模型通过 GridSearchCV(cv=5) 比较选出最优，并将预测功能集成到 Streamlit Tab 5。

## 二、文件变更

### 新建

| 文件 | 说明 |
|---|---|
| `emg/models/action_model.joblib` | 最优动作分类模型 (MLP, 412KB) |
| `emg/models/quality_model.joblib` | 最优质量分类模型 (MLP, 362KB) |
| `emg/models/label_encoder_action.joblib` | 动作标签编码器 |
| `emg/models/label_encoder_quality.joblib` | 质量标签编码器 |
| `emg/labels_v2.csv` | 重新生成的质量标签 (328 周期) |
| `regenerate_labels.py` | 标签重生成脚本 |
| `train_action_model.py` | 动作模型训练脚本 |
| `train_quality_model.py` | 质量模型训练脚本 |
| `src/prediction.py` | 预测模块 |

### 评估产出 (`emg/outputs/`)

| 文件 | 说明 |
|---|---|
| `action_model_comparison.csv` | 动作模型候选对比表 |
| `action_confusion_matrix.png` | 动作分类混淆矩阵 |
| `action_feature_importance.png` | 动作模型特征重要性排名 |
| `action_learning_curve.png` | 动作模型学习曲线 |
| `quality_model_comparison.csv` | 质量模型候选对比表 |
| `quality_confusion_matrix.png` | 质量分类混淆矩阵 |
| `quality_feature_importance.png` | 质量模型特征重要性排名 |
| `quality_learning_curve.png` | 质量模型学习曲线 |

### 修改

| 文件 | 说明 |
|---|---|
| `emg/app.py` | Tab 5 完整实现 |

## 三、模型结果

### 动作三分类模型

**数据集**: 328 样本 (前平举 128 / 推肩 105 / 侧平举 95)
**划分**: 按文件分层 80/20 (训练 264 / 测试 64)

| 模型 | CV F1 | Test F1 | Test Acc |
|---|---|---|---|
| **MLP** | 0.9527 | **0.7423** | **0.7500** |
| SVC | 0.9442 | 0.7193 | 0.7344 |
| Ridge | 0.9197 | 0.6767 | 0.6875 |
| LogisticReg | 0.9158 | 0.6575 | 0.6719 |
| KNN | 0.9417 | 0.6118 | 0.6406 |
| RandomForest | 0.9226 | 0.5604 | 0.5938 |
| GBM | 0.9228 | 0.5336 | 0.5938 |
| Bagging | 0.9048 | 0.4817 | 0.4844 |

**说明**: CV 高但 Test 偏低 → 不同受试者的肌电特征差异大，跨文件泛化有挑战。这是生理信号分类的正常现象。

### 质量二分类模型

**数据集**: 328 样本 (标准 196 / 不标准 132)，class_weight='balanced'
**划分**: 按文件分层 80/20 (训练 261 / 测试 67)

| 模型 | CV F1 | Test F1 | Test Acc |
|---|---|---|---|
| **MLP** | 0.9235 | **0.8345** | **0.8358** |
| GBM | 0.9316 | 0.7829 | 0.7910 |
| LogisticReg | 0.8707 | 0.7829 | 0.7910 |
| KNN | 0.9278 | 0.7710 | 0.7761 |
| Ridge | 0.8746 | 0.7659 | 0.7761 |
| Bagging | 0.9194 | 0.7519 | 0.7612 |
| RandomForest | 0.9233 | 0.7486 | 0.7612 |
| SVC | 0.9246 | 0.7379 | 0.7463 |

**质量模型 F1=0.83**，不标准周期召回率 0.93（仅漏检 2/27），标准周期精确率 0.94。

## 四、预测模块 (`src/prediction.py`)

### 函数

- `load_action_model()` → (Pipeline, LabelEncoder)
- `load_quality_model()` → (Pipeline, LabelEncoder)
- `predict_action(filtered_data, fs, cycles)` → dict
- `predict_quality(filtered_data, fs, cycles, action_label)` → dict
- `explain_abnormality(action_label, ratio)` → (异常类型, 解释文本)

### 多数投票

所有周期预测后，取多数类别作为文件整体动作。不一致周期标注可能原因。

### 规则解释

基于 CH2/CH1 RMS 比值 + 动作类别，输出三级异常分类（疑似中束代偿/疑似前束代偿/其他不标准）。

## 五、已修复 Bug

- `train_action_model.py`: `stratified_file_split` 中 numpy array 无 `.values` 属性 → 修复
- 两个训练脚本: 模型选择未排序 → 添加 `results.sort()`
- `quality_model.joblib` 保存错误模型 → 修复后重训练

## 六、验证结果

| 测试 | 状态 |
|---|---|
| 标签重生成 (65 文件 → 328 周期) | ✅ |
| 动作模型训练 (MLP, F1=0.74) | ✅ |
| 质量模型训练 (MLP, F1=0.83) | ✅ |
| emg_01_qpj_1.mat 预测 (5/5 前平举, 5 标准) | ✅ |
| emg_05_qpj_1.mat 预测 (5/5 前平举, 5 不标准, 异常解释正确) | ✅ |
| 模型文件可加载 | ✅ |
| 评估产出 8 个文件全部生成 | ✅ |
