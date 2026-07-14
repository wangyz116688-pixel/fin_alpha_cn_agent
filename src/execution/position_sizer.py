"""Position sizing utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PositionSizer:
    """Allocate capital to selected stocks and round volume to board lots."""

    def __init__(
        self,
        buffer_ratio: float = 0.1,
        max_single_ratio: float = 0.3,
        min_volume: int = 100,
        allocation_method: str = "score",
    ):
        self.buffer_ratio = buffer_ratio
        self.max_single_ratio = max_single_ratio
        self.min_volume = min_volume
        self.allocation_method = allocation_method

    def allocate(
        self,
        decisions: List[dict],
        available_capital: float = 500000.0,
        prev_closes: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        if not decisions:
            return []

        if prev_closes is None:
            prev_closes = {}

        for item in decisions:
            if "prev_close" not in item:
                item["prev_close"] = prev_closes.get(item["symbol"], 0.0)

        weights = self._weights(decisions)
        if not weights:
            return decisions

        effective_capital = available_capital * (1.0 - self.buffer_ratio)
        for item in decisions:
            weight = weights.get(item["symbol"], 0.0)
            amount = min(
                effective_capital * weight,
                available_capital * self.max_single_ratio,
            )

            prev_close = item.get("prev_close", 0.0)
            if prev_close <= 0:
                volume = 0
            else:
                volume = int(amount / prev_close / 100) * 100

            if volume < self.min_volume:
                volume = 0
                logger.warning(
                    "%s amount is too small to buy %d shares; skipped",
                    item["symbol"],
                    self.min_volume,
                )

            item["weight"] = round(weight, 4)
            item["amount"] = round(amount, 2)
            item["volume"] = volume

        return decisions

    def _weights(self, decisions: List[dict]) -> Dict[str, float]:
        method = (self.allocation_method or "score").lower()
        if method == "equal":
            return {item["symbol"]: 1.0 / len(decisions) for item in decisions}

        total_score = sum(item.get("C_mixed", 0.0) for item in decisions)
        if total_score <= 0:
            logger.warning("total confidence is 0; cannot allocate capital")
            return {}
        return {
            item["symbol"]: item.get("C_mixed", 0.0) / total_score
            for item in decisions
        }
