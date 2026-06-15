"""
资金分配模块。

按 C_mixed 加权分配资金，留 10% 缓冲，volume 向下取 100 整数倍。

对外接口:
    PositionSizer 类
        - allocate(decisions, available_capital, prev_closes) -> list[dict]
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    资金分配器。

    Parameters
    ----------
    buffer_ratio : float
        安全缓冲比例，默认 0.1（留 10%）。
    max_single_ratio : float
        单只股票最大占用资金比例，默认 0.3。
    min_volume : int
        最少持仓股数，不足则跳过。
    """

    def __init__(
        self,
        buffer_ratio: float = 0.1,
        max_single_ratio: float = 0.3,
        min_volume: int = 100,
    ):
        self.buffer_ratio = buffer_ratio
        self.max_single_ratio = max_single_ratio
        self.min_volume = min_volume

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def allocate(
        self,
        decisions: List[dict],
        available_capital: float = 500000.0,
        prev_closes: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """
        按 C_mixed 加权分配资金，计算每只股票的买入股数。

        Parameters
        ----------
        decisions : list[dict]
            候选决策列表，每项需含至少:
            - symbol: str
            - symbol_name: str
            - C_mixed: float (0~1)
        available_capital : float
            当日可用资金总额。
        prev_closes : dict, optional
            {symbol: 前收盘价}，若未提供则需在 decisions 中包含 prev_close。

        Returns
        -------
        list[dict]
            每项在原 decisions 基础上增加:
            - prev_close: float
            - weight: float
            - amount: float (买入金额)
            - volume: int (买入股数，100 整数倍)
        """
        if not decisions:
            return []

        if prev_closes is None:
            prev_closes = {}

        # 确保每项都有 prev_close
        for d in decisions:
            if "prev_close" not in d:
                d["prev_close"] = prev_closes.get(d["symbol"], 0.0)

        # 总置信度
        total_c = sum(d.get("C_mixed", 0.0) for d in decisions)
        if total_c <= 0:
            logger.warning("总置信度为 0，无法分配")
            return decisions

        # 可用资金（扣缓冲）
        effective_capital = available_capital * (1.0 - self.buffer_ratio)

        for d in decisions:
            c = d.get("C_mixed", 0.0)
            weight = c / total_c if total_c > 0 else 0.0
            amount = effective_capital * weight

            # 单只上限
            amount = min(amount, available_capital * self.max_single_ratio)

            prev_close = d.get("prev_close", 0.0)
            if prev_close <= 0:
                volume = 0
            else:
                volume = int(amount / prev_close / 100) * 100

            # 最少 100 股
            if volume < self.min_volume:
                volume = 0
                logger.warning("%s 买入金额不足以购买 %d 股，跳过", d["symbol"], self.min_volume)

            d["weight"] = round(weight, 4)
            d["amount"] = round(amount, 2)
            d["volume"] = volume

        return decisions