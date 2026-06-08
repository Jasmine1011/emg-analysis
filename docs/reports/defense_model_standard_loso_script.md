# 给动作做一次“健身房体检”：融合标准性参考的 sEMG 识别系统验证

> 展示口径：本部分只分析项目自有数据集；动作是否标准作为系统判断和结果解释的参考依据之一，其中分类模型的 LOSO 验证使用 `labels_v2.csv` 中 `quality_label == "标准"` 的动作周期；不纳入 Group03 隐藏测试集，也不纳入文件夹 2 的补充数据。

## 讲述主线

这部分我不从“我们试了很多模型”开始讲，而是从验证问题本身开始：如果模型将来要面对一个没参与训练的新同学，它能不能只凭双通道 sEMG 的周期特征识别前平举、侧平举和推肩？因此我们采用 LOSO，也就是每次留出一个受试者，其余受试者训练，最后把所有留出结果合并计算 Precision、Recall、F1 和混淆矩阵。

## 1. 数据口径先固定：只看标准动作，避免质量差异干扰动作分类

- 标准周期总数：**174**
- 受试者数：**13**
- 动作分布：前平举 70，侧平举 42，推肩 62

![标准样本分布](F:/Project_final/outputs/standard_loso_assets/standard_data_distribution.png)

讲稿：这里我先把分类问题收窄到“标准动作之间怎么分”。这样做的原因是，如果把不标准动作也放进动作分类，模型可能学到的是动作质量或代偿模式，而不是动作类别本身。标准样本的分布并不是完全均衡，这也是为什么后面统一使用 macro Precision、macro Recall 和 macro F1，而不是只看 accuracy。

## 2. 验证方式选择 LOSO，因为它更接近真实泛化场景

LOSO 的含义是 Leave-One-Subject-Out：每一轮把一个受试者完全留作测试，其余受试者训练。这个验证方式比随机划分更严格，因为随机划分可能让同一个人的不同周期同时出现在训练集和测试集里，指标会偏乐观。我们的目标是验证跨人的动作识别，所以 LOSO 更符合任务风险。


![逐受试者 LOSO 表现](F:/Project_final/outputs/standard_loso_assets/standard_loso_by_subject.png)

说明：逐受试者柱状图的 F1 只在该受试者实际出现过的动作类别上计算；最终模型对比仍以所有 LOSO 测试折合并后的 macro Precision、Recall、F1 为主指标。

## 3. 模型选择：用同一套 LOSO 口径比较，而不是只看训练集拟合

在当前最优特征集下，不同模型的 LOSO 结果如下：

| model              |   precision_macro |   recall_macro |   f1_macro |   accuracy |
|:-------------------|------------------:|---------------:|-----------:|-----------:|
| LogisticRegression |             0.985 |          0.985 |      0.985 |      0.983 |
| RBF-SVM            |             0.980 |          0.981 |      0.980 |      0.977 |
| KNN                |             0.975 |          0.976 |      0.975 |      0.971 |
| ExtraTrees         |             0.975 |          0.976 |      0.975 |      0.971 |
| RidgeClassifier    |             0.975 |          0.973 |      0.973 |      0.971 |
| RandomForest       |             0.969 |          0.970 |      0.970 |      0.966 |

![模型比较](F:/Project_final/outputs/standard_loso_assets/model_comparison_best_feature_set.png)

结论：本轮只看自有标准数据时，最优组合是 **LogisticRegression + 15个稳健特征**，LOSO macro Precision = **0.985**，Recall = **0.985**，F1 = **0.985**。
模型选择理由不是“模型越复杂越好”，而是看跨受试者留出时能否稳定识别三类动作。树模型能处理非线性边界，线性模型更稳定且可解释，SVM/KNN 对小样本边界敏感；最终选择取决于 LOSO 下的 macro F1 和错误模式，而不是单次随机划分。

## 4. 特征选择：文献基础特征是底座，双通道协同特征解释动作差异

我们比较了四组特征：文献基础特征、协同/比值特征、15 个稳健特征、全部候选特征。每组都用相同 LOSO 口径选出该组下表现最好的模型：

| feature_set   |   n_features | model              |   precision_macro |   recall_macro |   f1_macro |
|:--------------|-------------:|:-------------------|------------------:|---------------:|-----------:|
| 15个稳健特征  |           15 | LogisticRegression |             0.985 |          0.985 |      0.985 |
| 全部候选特征  |           60 | RandomForest       |             0.980 |          0.981 |      0.980 |
| 文献基础特征  |           28 | RandomForest       |             0.980 |          0.981 |      0.980 |
| 协同/比值特征 |           49 | RBF-SVM            |             0.862 |          0.861 |      0.859 |

![特征集比较](F:/Project_final/outputs/standard_loso_assets/feature_set_comparison.png)

讲稿：RMS、MAV、WL、ZC、SSC、MF/MDF 这类特征来自常见 sEMG 分类工作，分别描述激活强度、波形复杂度和频谱结构。但我们的任务只有两个通道，而且动作差异和三角肌前束/中束的相对激活有关，所以不能只看单通道绝对幅值。CH2/CH1 比值、RMS 主导度、通道峰值时序等协同特征能把“哪块肌肉更主导”显式表达出来，这就是特征选择的核心理由。

## 5. 混淆矩阵：分类瓶颈要从哪几类互相混淆里看

|           |   预 前平举 |   预 侧平举 |   预 推肩 |
|:----------|------------:|------------:|----------:|
| 真 前平举 |          68 |           0 |         2 |
| 真 侧平举 |           0 |          42 |         0 |
| 真 推肩   |           1 |           0 |        61 |

![LOSO 混淆矩阵](F:/Project_final/outputs/standard_loso_assets/standard_loso_confusion_matrix.png)

讲稿：混淆矩阵比单个 F1 更有解释价值。如果侧平举错误少，说明 CH2/CH1 这类中束主导特征是有效的；如果推肩和前平举互相混淆，就说明仅靠当前双通道周期统计还不能完全描述推肩这种复合动作，需要进一步加入周期内形态或更多肌肉通道信息。

## 6. 特征重要性：模型真正依赖的是幅值、比值和主导关系

| feature   |   importance_mean |   importance_std |
|:----------|------------------:|-----------------:|
| var_ratio |            0.0766 |           0.1106 |
| diff_rms  |            0.0371 |           0.0719 |
| mav_ch1   |            0.0225 |           0.0465 |
| iemg_ch1  |            0.0212 |           0.0492 |
| rms_ch1   |            0.0207 |           0.0467 |
| var_ch1   |            0.0189 |           0.0367 |
| ratio_rms |            0.0062 |           0.0232 |
| rms_ch2   |            0.0040 |           0.0108 |
| iemg_dom  |            0.0000 |           0.0000 |
| mav_dom   |            0.0000 |           0.0000 |
| var_ch2   |            0.0000 |           0.0000 |
| rms_dom   |           -0.0011 |           0.0039 |

![特征重要性](F:/Project_final/outputs/standard_loso_assets/standard_loso_feature_importance.png)

讲稿：这里的特征重要性使用 LOSO 测试折上的 permutation importance：也就是每次打乱一个特征，看 Macro F1 下降多少。越靠前的特征，说明它对跨受试者分类越关键。这个结果可以和生理解释对应起来：三角肌前束和中束的绝对激活强度提供动作负荷信息，而比值/主导度特征提供肌肉协同信息。

## 7. 可以这样收束这一页模型分析

这一部分的结论是：在我们自己的标准数据上，动作分类并不是单纯靠一个黑箱模型完成的。验证方式上，我们用 LOSO 避免同一受试者泄漏；模型选择上，我们用 macro F1 和混淆矩阵判断跨人泛化；特征选择上，我们把常见 EMG 特征作为底座，再加入符合肩部动作生理的双通道协同特征。这个结果说明当前系统已经能在标准动作上建立可解释的分类边界，但推肩这类复合动作仍然是后续改进重点。

## 备用数据：分类报告

```text
              precision    recall  f1-score   support

         前平举      0.986     0.971     0.978        70
         侧平举      1.000     1.000     1.000        42
          推肩      0.968     0.984     0.976        62

    accuracy                          0.983       174
   macro avg      0.985     0.985     0.985       174
weighted avg      0.983     0.983     0.983       174
```

## 参考依据

- SENIAM 关于表面肌电传感器位置和固定方式的建议强调：肌肉对应位置、双极电极、方向与间距会影响 sEMG 采集质量，因此本项目把两个通道分别对应三角肌前束和中束。[SENIAM placement/fixation](https://seniam.org/fixation.htm)
- sEMG 分类综述中常见特征包括 MAV、RMS、WL、ZC、SSC、IEMG、MF/MDF 等，这支持我们用时域、频域和波形复杂度特征作为基础。[Surface EMG Signal Processing and Classification Techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC3821366/)
- 时间域 EMG 特征稳定性研究解释了 MAV、ZC、SSC 等特征在模式识别中的定义和作用，因此这些特征适合作为基础候选特征。[Study of stability of time-domain features for EMG pattern recognition](https://jneuroengrehab.biomedcentral.com/articles/10.1186/1743-0003-7-21)
