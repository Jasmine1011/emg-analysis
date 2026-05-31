# EMG Analysis Platform

双通道表面肌电信号处理与分析系统

**在线体验**: [待部署] Streamlit Community Cloud

## 功能
- 上传 .mat 文件，自动完成 EMG 信号预处理、动作事件检测、特征提取
- 动作类型三分类识别：前平举 / 侧平举 / 推肩
- 动作质量辅助判断（标准 / 不标准 + 异常解释）

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## 项目结构
```
├── app/                     # Streamlit 应用
├── src/                     # 核心算法模块
├── scripts/                 # 训练 & 标签脚本
├── data/                    # 数据 & 标签
├── models/                  # 训练好的模型
├── outputs/                 # 评估结果
└── docs/                    # 文档
```

## 技术栈
Python · Streamlit · NumPy · SciPy · scikit-learn · Matplotlib · Pandas

## 免责声明
⚠️ 动作质量判断为辅助参考，不作为严格医学或运动学评价结果
