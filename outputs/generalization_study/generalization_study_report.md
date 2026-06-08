# Group03 泛化能力研究报告

## 研究口径

- 训练数据限定为 `data/` 与 `2/`。
- `Group03/True_Labels_map.csv` 只用于最终评估和误差分析，读取第 1 列 ID 与第 3 列映射标签。
- 本轮不加入深度学习，重点比较训练数据来源、特征组、模型复杂度、文件级决策方式和简单规则/聚合模型。

## 关键结果

最佳配置为 `data_plus_2_auto / stable15_plus_timing / SoftVote_LR_SVM_RF / mean_proba`，Group03 Macro-F1 = **0.795**，Precision = **0.857**，Recall = **0.806**，Accuracy = **0.806**。
当前稳健 LR 基线附近的最好 Group03 Macro-F1 约为 **0.705**。
真实分布：`{1: 12, 2: 12, 3: 12}`；最佳配置预测分布：`{1: 17, 2: 13, 3: 6}`。

### Top 10 配置

| config                                                                       |   precision_macro |   recall_macro |   f1_macro |   accuracy |   loso_file_f1 |
|:-----------------------------------------------------------------------------|------------------:|---------------:|-----------:|-----------:|---------------:|
| data_plus_2_auto / stable15_plus_timing / SoftVote_LR_SVM_RF / mean_proba    |          0.856712 |       0.805556 |   0.795096 |   0.805556 |       0.728465 |
| data_auto / stable15_plus_timing / SoftVote_LR_SVM_RF / mean_proba           |          0.866667 |       0.777778 |   0.775253 |   0.777778 |       0.736543 |
| data_auto / stable15_plus_timing / Ridge / majority_vote                     |          0.842593 |       0.777778 |   0.772222 |   0.777778 |       0.690507 |
| data_plus_2_auto / stable15_plus_timing / LR_C0.1 / mean_proba               |          0.842593 |       0.777778 |   0.772222 |   0.777778 |     nan        |
| data_auto / stable15_plus_timing / LinearSVM / majority_vote                 |          0.84689  |       0.777778 |   0.770142 |   0.777778 |     nan        |
| data_plus_2_auto / stable15_plus_timing / SoftVote_LR_SVM_RF / majority_vote |          0.827381 |       0.777778 |   0.76801  |   0.777778 |     nan        |
| data_plus_2_auto / stable15_plus_timing / Ridge / majority_vote              |          0.827381 |       0.777778 |   0.76801  |   0.777778 |       0.686686 |
| data_auto / stable15_plus_timing / SoftVote_LR_SVM_RF / majority_vote        |          0.857143 |       0.75     |   0.750361 |   0.75     |     nan        |
| data_plus_2_auto / stable15_plus_timing / LinearSVM / majority_vote          |          0.803571 |       0.75     |   0.742369 |   0.75     |     nan        |
| data_plus_2_auto / stable15_plus_timing / LR_C0.03 / majority_vote           |          0.857143 |       0.75     |   0.741533 |   0.75     |       0.689181 |

### 最佳配置混淆矩阵

|        |   前平举 |   侧平举 |   推肩 |
|:-------|---------:|---------:|-------:|
| 前平举 |       11 |        1 |      0 |
| 侧平举 |        0 |       12 |      0 |
| 推肩   |        6 |        0 |      6 |

## 三点发现

### 发现 1：训练集内部高分不能直接代表隐藏集泛化

多数组合在训练集 LOSO 上明显高于 Group03 表现，说明主要风险不是模型在自有数据上学不会，而是跨采集条件、跨受试者和动作执行习惯变化后，原有边界发生偏移。
建议展示图：`outputs/generalization_study/loso_vs_group03_gap.png`。

### 发现 2：特征不是越多越好，低漂移的稳健特征更关键

以 `ratio_rms` 为例，训练集类别均值为 `{'前平举': 0.724, '侧平举': 1.599, '推肩': 0.664}`，Group03 类别均值为 `{'前平举': 0.993, '侧平举': 1.179, '推肩': 0.545}`。
幅值、比值、相关性和时长类特征在 Group03 上都出现不同程度分布偏移。因此全量堆叠特征容易把采集差异也学进去；更稳的做法是选择 LOSO 稳定且跨域漂移较小的特征。
建议展示图：`outputs/generalization_study/train_vs_group03_feature_shift.png` 与 `feature_model_heatmap.png`。

### 发现 3：文件级决策层会放大或修正周期级错误

周期级模型最终要转成文件级动作标签。多数投票、概率平均、文件级聚合模型得到的结果不同，说明泛化不只由分类器决定，还由周期检测质量、每个周期的置信度分布和文件级融合策略共同决定。
建议展示图：`outputs/generalization_study/decision_strategy_comparison.png`。

## 错误案例

| filename   | true_action   | pred_action   |   cycles |   confidence |   proba_margin | cycle_votes   |
|:-----------|:--------------|:--------------|---------:|-------------:|---------------:|:--------------|
| 3.mat      | 前平举        | 侧平举        |        6 |     0.578838 |       0.174884 | {2: 6}        |
| 7.mat      | 推肩          | 前平举        |        5 |     0.690077 |       0.394468 | {1: 4, 2: 1}  |
| 9.mat      | 推肩          | 前平举        |        6 |     0.785489 |       0.588915 | {1: 6}        |
| 18.mat     | 推肩          | 前平举        |        7 |     0.699158 |       0.414294 | {1: 7}        |
| 23.mat     | 推肩          | 前平举        |        7 |     0.60755  |       0.230335 | {1: 6, 2: 1}  |
| 31.mat     | 推肩          | 前平举        |        7 |     0.552456 |       0.121537 | {1: 5, 2: 2}  |
| 36.mat     | 推肩          | 前平举        |        8 |     0.57733  |       0.168746 | {1: 8}        |

## 图表清单

- `outputs/generalization_study/top10_group03_f1.png`
- `outputs/generalization_study/best_confusion_matrix.png`
- `outputs/generalization_study/feature_model_heatmap.png`
- `outputs/generalization_study/loso_vs_group03_gap.png`
- `outputs/generalization_study/train_vs_group03_feature_shift.png`
- `outputs/generalization_study/decision_strategy_comparison.png`
- `outputs/generalization_study/model_complexity_vs_generalization_research.png`
