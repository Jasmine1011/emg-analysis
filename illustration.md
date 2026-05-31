# project\_final

# 项目定位

**一个可本地运行的 Streamlit 肌电信号处理与动作识别 Demo 系统**，实现 `\.mat` 文件上传、双通道 EMG 预处理、动作事件检测、特征提取、动作分类、动作质量辅助判断与可视化展示。

# 一、技术路线

```Plain Text
MAT 文件数据
    ↓
Python 读取 Fs 和 data
    ↓
双通道 EMG 预处理
    ↓
RMS 包络计算
    ↓
动作事件检测，分割 4–6 个完整周期
    ↓
每个周期提取特征
    ↓
动作类型三分类模型
    ↓
动作质量二分类模型
    ↓
不标准样本进入规则型异常解释
    ↓
Streamlit 网页可视化展示
```

技术栈全部使用免费工具：

```Plain Text
Python
Streamlit
NumPy
SciPy
Pandas
Matplotlib
scikit-learn
joblib
```

本地演示输入`streamlit run app\.py`，浏览器会自动打开本地网页

# 二、数据组织方式

`\.mat` 文件格式

```Plain Text
Fs      -> double，采样率，2000 Hz
data    -> N × 2 double 矩阵
          第 1 列：三角肌前束
          第 2 列：三角肌中束
```

文件命名规则：

```Plain Text
emg_01_cpj_1.mat
```

含义：

```Plain Text
01  -> 受试者编号
cpj -> 侧平举
qpj -> 前平举
tj  -> 推肩
1   -> 正常状态
2   -> 疲劳状态
```

动作分类训练时，`\_1` 和 `\_2` 都参与训练，不区分疲劳，只根据 `cpj/qpj/tj` 识别动作类别。

需要建立一个最简标签表 `labels\.csv`，填写质量标签：

```Plain Text
filename,quality_label
emg_01_cpj_1.mat,标准
emg_01_cpj_2.mat,标准
emg_02_qpj_1.mat,不标准
emg_02_qpj_2.mat,不标准
```

程序自动从文件名解析：

```Plain Text
subject_id
action_label
state
```

# 三、预处理模块设计

```Plain Text
采样率 Fs：从 mat["Fs"] 读取，理论为 2000 Hz
幅值换算：data * (0.286 / 6)
去直流：detrend 或减均值
高通滤波：20 Hz，4 阶 Butterworth
低通滤波：450 Hz，4 阶或 6 阶 Butterworth
工频陷波：50 Hz 及其谐波
滤波方式：scipy.signal.filtfilt 零相位滤波
```

网页中预处理部分保留参数控件：

```Plain Text
是否去直流
是否陷波
工频频率：默认 50 Hz
高通截止频率：默认 20 Hz
低通截止频率：默认 450 Hz
是否进行幅值换算
```

点击“开始预处理”后展示：

```Plain Text
CH1 原始时域图 vs 处理后时域图
CH1 原始频域图 vs 处理后频域图
CH2 原始时域图 vs 处理后时域图
CH2 原始频域图 vs 处理后频域图
```

# 四、动作事件检测方案

考虑采用

> **RMS 包络 \+ 平滑 \+ 自适应阈值 \+ 最小周期时长约束 \+ 峰/谷边界修正**

具体流程：

```Plain Text
1. 对两个通道分别计算短窗 RMS 包络
2. 将两个通道包络归一化
3. 得到融合包络：
   envelope = 0.5 * envelope_ch1 + 0.5 * envelope_ch2
4. 对 envelope 做平滑
5. 用自适应阈值识别活动区间
6. 合并过短间隔
7. 删除过短或过长的片段
8. 输出每个完整动作周期的起止时间
```

默认参数建议：

```Plain Text
RMS 窗口：200 ms
重叠率：50%
平滑窗口：300–500 ms
单次动作最短时长：1.5 s
单次动作最长时长：8 s
默认参考动作时长：4 s
最小静息间隔：0.3–0.5 s
```

网页演示时允许调参：

```Plain Text
阈值比例
平滑窗口
最小周期时长
最大周期时长
最小间隔
```

但正式训练和测试时使用默认参数全自动运行。

展示结果：

```Plain Text
检测到完整动作次数：5 次
周期 1：0.8–4.7 s
周期 2：5.2–9.1 s
……
```

并在预处理后时域图上画出每个周期边界

# 五、特征提取方案

每个完整运动周期作为一个训练样本。也就是说，70 个 `\.mat` 文件如果每个包含 4–6 次动作，大约可以得到 280–420 个周期样本，每个周期对两个通道分别提取特征。

```Plain Text
时域特征：
RMS
MAV
VAR
WL
ZC
SSC
IEMG

频域特征：
MF
MDF
PF

双通道协同特征：
CH2/CH1 RMS 比值
CH1 与 CH2 RMS 差值
CH1 与 CH2 RMS 占比
双通道相关系数
CH1 峰值时间
CH2 峰值时间
CH2-CH1 激活时间差
```

其中：

```Plain Text
CH1 = 三角肌前束
CH2 = 三角肌中束
```

网页中特征提取模块默认勾选：

```Plain Text
RMS
MAV
WL
MF
MDF
CH2/CH1 RMS 比值
双通道相关系数
```

展示方式：

```Plain Text
顶部：预处理后 EMG 时域图
下面：RMS 曲线
下面：MF 曲线
下面：CH2/CH1 RMS 比值曲线
下面：特征表格
```

# 六、动作类型识别模型

动作类型模型做三分类：

```Plain Text
前平举
侧平举
推肩
```

训练方式：

```Plain Text
输入：每个动作周期的特征向量
标签：从文件名中的 qpj/cpj/tj 自动解析
模型：传统机器学习模型
```

AI建议模型优先级：

```Plain Text
首选：Random Forest
备选：SVM
对照：Logistic Regression
```

主模型用 **Random Forest**，原因是：

```Plain Text
1. 不需要深度学习，适合小样本
2. 对特征尺度不太敏感
3. 能处理非线性关系
4. 可输出特征重要性，便于报告解释
5. 答辩时容易讲清楚
```

训练/测试划分：

```Plain Text
按 .mat 文件随机划分
80% 文件作为训练集
20% 文件作为测试集
```

测试时流程：

```Plain Text
一个测试 .mat 文件
    ↓
自动分割成多个周期
    ↓
每个周期分别预测动作类别
    ↓
多数投票得到整个文件的动作类别
```

网页输出示例：

```Plain Text
整体疑似动作：前平举

周期分类结果：
周期 1：前平举
周期 2：前平举
周期 3：前平举
周期 4：侧平举
周期 5：前平举

整体判断依据：
5 个完整周期中，4 个周期被分类为前平举，因此整体判断为前平举。
第 4 个周期分类结果不同，可能与动作不标准、肌肉代偿或事件分割误差有关。
```

# 七、动作质量判断方案

采用两级结构：

```Plain Text
第一级：机器学习二分类
标准 / 不标准

第二级：规则型辅助解释
疑似中束代偿 / 疑似前束代偿 / 其他不标准
```

网页上注明：

> **动作质量判断为辅助判断，不作为严格医学或运动学评价结果。**
> 
> 

质量二分类模型：

```Plain Text
输入：周期特征
标签：labels.csv 中的 标准 / 不标准
```

由于你们采集时“不标准文件整段都是不标准”，因此可以把文件级质量标签赋给该文件内所有周期：

```Plain Text
emg_02_qpj_1.mat -> 不标准
周期 1 -> 不标准
周期 2 -> 不标准
周期 3 -> 不标准
```

不标准后的规则解释：

```Plain Text
如果 CH2/CH1 RMS 比值明显高于该动作标准模板范围：
    疑似中束代偿

如果 CH1/CH2 RMS 比值明显高于该动作标准模板范围：
    疑似前束代偿

如果双通道激活比例异常但不符合上述两类：
    其他不标准

如果周期长度异常、包络多峰、周期内激活不稳定：
    其他不标准或动作执行不稳定
```

标准模板范围可以从训练集中“标准样本”统计得到：

```Plain Text
每个动作类别分别计算：
CH2/CH1 RMS 比值的均值 ± 标准差
双通道相关系数范围
激活时间差范围
RMS 变异系数范围
```

例如：

```Plain Text
当前周期被质量模型判断为“不标准”。

主要异常：疑似中束代偿

解释：
该周期的中束/前束 RMS 比值高于前平举标准样本范围，且中束激活峰值提前出现，说明动作可能存在向外展方向偏移或三角肌中束代偿。
```

# 八、Streamlit 网页结构

## 模块 1：项目说明

显示：

```Plain Text
项目名称
双通道 EMG 动作识别系统
通道说明
支持动作：前平举、侧平举、推肩
提示：动作质量判断为辅助判断
```

## 模块 2：数据上传

功能：

```Plain Text
上传 1–10 个 .mat 文件
读取 Fs 和 data
显示文件名、采样率、数据长度、通道数
```

界面参考：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTY5MGZjZGU5ZjRiOTBkM2ZmMDEyZTkyZTNkYmI0MmNfYTZiY2M1NzYyYjgwMTE2NjdiMTNlZDgyN2Q3NGVhZjVfSUQ6NzY0NTIyNTA5ODkxMjA3NDcwMF8xNzgwMjM1MzA0OjE3ODAzMjE3MDRfVjM)

## 模块 3：预处理

功能：

```Plain Text
参数选择
开始预处理按钮
显示双通道处理前后时域/频域图
```

界面参考：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjNmOGY5NzAxZmUzZjg3YWQxZjhiZjRlZTUzMTNmYmZfNmQzY2ZhNmE5MmRmZTcxYjY0ZTJiOTc2YTdmN2Q5ODhfSUQ6NzY0NTIyNTIzNjA3ODQ0NzU4OF8xNzgwMjM1MzA0OjE3ODAzMjE3MDRfVjM)

## 模块 4：动作事件检测

功能：

```Plain Text
显示 RMS 融合包络
显示检测到的周期数量
在时域图上标出周期边界
展示每个周期起止时间
```

标注参考：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MWFiNTgyNTkwYjRhNWU1OTc1Y2ViOGRmMGVhZDljN2JfNDVjOGMzZjlmNzY5ZDI5NWNhYjZmMGVmNDE4MTY1ZjNfSUQ6NzY0NTIyNTcyMTY2NTk0ODYxOV8xNzgwMjM1MzA0OjE3ODAzMjE3MDRfVjM)

## 模块 5：特征提取

功能：

```Plain Text
选择特征
显示特征曲线
显示周期级特征表
```

特征曲线参考：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWE4M2Y2N2I3YTczMzk0MGZjNWFiYjMxMGYzNTY4M2ZfNDI2OGY5M2YzODc1ZTk0NjQyOTlhMWIwN2NjZTZkMjlfSUQ6NzY0NTIyNTU1Nzk2MzQyNjc0Nl8xNzgwMjM1MzA0OjE3ODAzMjE3MDRfVjM)

## 模块 6：动作分类与质量判断

功能：

```Plain Text
每个周期动作分类
整体动作投票
每个周期质量判断
整体质量判断
不标准周期异常解释
```

# 九、代码文件结构（AI建议，仅供参考）

建议项目文件夹这样组织：

```Plain Text
emg_project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── labels.csv
│
├── data/
│   ├── emg_01_cpj_1.mat
│   ├── emg_01_cpj_2.mat
│   └── ...
│
├── models/
│   ├── action_model.joblib
│   ├── quality_model.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
│
├── src/
│   ├── io_utils.py
│   ├── preprocessing.py
│   ├── event_detection.py
│   ├── features.py
│   ├── train_utils.py
│   ├── prediction.py
│   └── quality_rules.py
│
└── outputs/
    ├── metrics_action.csv
    ├── metrics_quality.csv
    └── feature_table.csv
```

每个文件职责：

```Plain Text
io_utils.py
读取 .mat 文件，解析文件名，读取 labels.csv

preprocessing.py
完成单位换算、去直流、陷波、高通、低通

event_detection.py
计算 RMS 包络，检测周期边界

features.py
提取周期级特征

train_model.py
训练动作三分类模型和质量二分类模型

prediction.py
加载模型，对新 .mat 文件预测

quality_rules.py
对不标准周期做代偿类型解释

app.py
Streamlit 网页主程序
```

---

# 十、开发顺序（AI建议，仅供参考）

## 第 1 步：完成 Python 版单文件处理

目标：

```Plain Text
读取一个 .mat
完成预处理
画出双通道时域/频域图
```

对应 MATLAB 代码迁移。

## 第 2 步：完成动作事件检测

目标：

```Plain Text
输入一个预处理后信号
输出 4–6 个周期边界
画图检查分割是否合理
```

## 第 3 步：完成周期特征提取

目标：

```Plain Text
每个周期输出一行特征
生成 feature_table.csv
```

## 第 4 步：完成训练脚本

目标：

```Plain Text
读取全部训练文件
按文件划分训练/测试集
分割周期
提取特征
训练动作分类模型
训练质量二分类模型
保存模型
输出准确率、混淆矩阵
```

## 第 5 步：完成 Streamlit 网页

目标：

```Plain Text
上传 .mat
加载已训练模型
展示预处理、检测、特征、分类、质量判断
```

## 第 6 步：整理报告和 PPT 逻辑

重点展示：

```Plain Text
为什么这样预处理
为什么用周期级样本
为什么用 Random Forest
动作识别结果如何
不标准动作如何辅助解释
误差来源是什么
```

---

# 十一、项目报告中的核心逻辑

本项目针对双通道表面肌电信号，构建了从信号预处理、动作事件检测、周期级特征提取到运动模式识别的完整流程。首先对原始 EMG 信号进行去直流、工频及谐波陷波、高通和低通滤波，以降低基线漂移、工频干扰、心电干扰和高频噪声。随后基于双通道 RMS 融合包络进行动作事件检测，自动识别完整运动周期，并对双通道信号进行同步分割。在每个动作周期内提取时域、频域和双通道协同特征，构建周期级特征样本。最后采用传统机器学习模型完成前平举、侧平举和推肩三类动作识别，并通过周期级预测结果的多数投票得到整段文件的动作类别。对于动作质量，本项目采用“机器学习二分类 \+ 规则型异常解释”的两级策略，先判断标准/不标准，再根据双通道激活比例和激活时序辅助解释可能存在的中束代偿、前束代偿或其他异常。

---

# 十二、其他

质量判断、疲劳分析、能量消耗等可以作为拓展模块，有空再展开做

