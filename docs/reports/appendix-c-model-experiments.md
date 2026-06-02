# 附录 C：模型选择与数据集优化实验

## C.1 备选模型完整对比

使用 47 维特征 + GroupKFold(cv=5) 评估：

| 模型 | 最佳参数 | CV F1 | 备注 |
|---|---|---|---|
| KNN (k=3, dist, manhattan) | MinMaxScaler | 0.90 | **最优** |
| KNN (k=5, dist, manhattan) | MinMaxScaler | 0.88 | |
| KNN (k=7, uniform, manhattan) | MinMaxScaler | 0.88 | |
| KNN (k=3, dist, cosine) | MinMaxScaler | 0.87 | 余弦距离劣于曼哈顿 |
| SVC (rbf, C=10) | MinMaxScaler | 0.87 | |
| SVC (poly, degree=3, C=10) | MinMaxScaler | 0.87 | 三次 SVM 曾被用户报告有效 |
| RandomForest (500 trees) | MinMaxScaler | 0.86 | |
| GradientBoosting (200, depth=3) | StandardScaler | 0.85 | |
| MLP (50 neurons) | MinMaxScaler | 0.82 | |
| Voting(KNN+GBM+RF, soft) | MinMaxScaler | 0.87 | 集成未超越最优单模型 |
| LogisticRegression | MinMaxScaler | 0.78 | |
| RidgeClassifier | MinMaxScaler | 0.81 | |

## C.2 为什么 Manhattan > Euclidean > Cosine

sEMG 的比率特征在特征空间中形成**轴对齐的簇**（CH1 为主导的特征在某一方向聚集，CH2 为主导的在另一方向）。Manhattan 距离（L1 范数）对轴对齐的分布更敏感——它沿坐标轴测量距离，而非最短直线距离。这恰好适合"某维度的值是否落在对应动作的典型范围内"这种 sEMG 分类场景。

## C.3 KNN 的 k 值影响

| k | weights | CV F1 |
|---|---|---|
| 3 | distance | **0.90** |
| 3 | uniform | 0.89 |
| 5 | distance | 0.88 |
| 7 | uniform | 0.88 |
| 9 | distance | 0.87 |

k 越小→决策边界越精细→但过拟合风险越大。k=3 在当前 174 样本上取得了最优平衡。Distance weighting 始终优于 uniform——近邻的贡献应该更大，这在受试者间特征分布差异大时尤为重要。

## C.4 数据集策略完整对比（LOSO-CV）

| 策略 | 样本数 | 受试者 | LOSO F1 |
|---|---|---|---|
| A: 全部样本 | 295 | 12 | 0.74 |
| B: 排除对照受试者(10,12) | 235 | 10 | 0.67 |
| C: 仅标准样本 | 174 | 10 | **0.92** |
| D: 仅标准+排除对照 | 174 | 10 | **0.92** |

策略 C 和 D 等效——因为对照受试者（10, 12）的所有样本本身就是不标准的，排除不标准自然排除了他们。

## C.5 受试者 04 专题分析

受试者 04 在所有版本中均表现最差（F1=0.40）。深入分析：

- 04 的侧平举（cpj）文件全部被标注为"不标准——疑似前束代偿"
- 其"标准"前平举和推肩的正常执行受其代偿策略影响
- 04 的 RMS 比值落在前平举和侧平举的临界区间（0.8–1.0）

这是**生理学本质限制**——个体的肌肉募集策略决定了 EMG 模式，当一个人的"标准动作"恰好落在两个类别的边界上时，任何基于 EMG 特征的分类器都会出现高错误率。这不是算法缺陷，而是 EMG 信号的个体差异天花板。
