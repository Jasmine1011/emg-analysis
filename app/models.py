# -*- coding: utf-8 -*-
"""
app/models.py — 数据模型

FileResult: 单个 .mat 文件的全生命周期分析结果
  阶段: raw → filtered → events → curves → features → action → quality
  每个阶段的结果以属性形式存储，替代嵌套字典。
"""

from dataclasses import dataclass, field


@dataclass
class FileResult:
    """单个 EMG 文件的分析结果"""
    fname: str
    # 原始数据
    raw: "np.ndarray" = None
    fs: float = 2000.0
    # 预处理
    filtered: "np.ndarray" = None
    # 事件检测
    events: dict = None
    # 特征
    curves: dict = None
    features_df: "pd.DataFrame" = None
    # 动作分类
    action: str = ""
    action_votes: dict = field(default_factory=dict)
    action_cycles: list = field(default_factory=list)
    # 质量评估
    std_count: int = 0
    nonstd_count: int = 0
    quality_cycles: list = field(default_factory=list)
    # 错误状态
    error: str = ""

    # ---- computed properties ----
    @property
    def is_done(self):
        """是否至少完成了预处理"""
        return self.filtered is not None

    @property
    def is_ok(self):
        """是否无错误"""
        return self.is_done and not self.error

    @property
    def cycle_count(self):
        if self.events:
            return self.events.get("count", 0)
        return 0

    @property
    def active_segments(self):
        if self.events:
            return self.events.get("active_segments", [])
        return []

    @property
    def rest_segments(self):
        if self.events:
            return self.events.get("rest_segments", [])
        return []

    @property
    def envelope(self):
        if self.events:
            return self.events.get("envelope")
        return None

    @property
    def env_ch1(self):
        if self.events:
            return self.events.get("envelope_ch1")
        return None

    @property
    def env_ch2(self):
        if self.events:
            return self.events.get("envelope_ch2")
        return None

    @property
    def baseline(self):
        if self.events:
            return self.events.get("baseline", 0)
        return 0

    @property
    def threshold(self):
        if self.events:
            return self.events.get("threshold", 0)
        return 0

    @property
    def segment_data(self):
        if self.events:
            return self.events.get("segment_data")
        return None

    def set_error(self, msg):
        """设置错误信息并标记"""
        self.error = str(msg)[:120]
