"""回测净值指标，统一包含建仓前的初始资金基准。"""

import numpy as np


def maximum_drawdown(nav, initial_nav: float = 1.0) -> float:
    """返回非正的最大回撤；无观察值返回NaN，净值按时间顺序输入。"""
    values = np.asarray(nav, dtype=float)
    if values.ndim != 1:
        raise ValueError("净值必须是一维时间序列")
    initial_nav = float(initial_nav)
    if not np.isfinite(initial_nav) or initial_nav <= 0:
        raise ValueError("初始净值必须为有限正数")
    if not np.isfinite(values).all():
        raise ValueError("净值不能包含缺失值或无穷值")
    if values.size == 0:
        return float("nan")
    # 即使首期亏损，初始资金仍是此前达到过的高点。
    path = np.concatenate(([initial_nav], values))
    peaks = np.maximum.accumulate(path)
    return float(np.min(path / peaks - 1.))
