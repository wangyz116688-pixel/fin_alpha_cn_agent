"""
比赛主入口。

串联全部模块：量化选股 → 多智能体决策 → 资金分配 → 输出 JSON。

使用方式:
    python -m src.main_stock_agent [--date YYYYMMDD] [--top-n 20] [--capital 500000]
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.models.stock_screener import StockScreener
from src.execution.position_sizer import PositionSizer
from src.execution.output_formatter import OutputFormatter

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaAgent 比赛主流程")
    parser.add_argument("--date", type=str, default=None, help="交易日期 YYYYMMDD")
    parser.add_argument("--top-n", type=int, default=20, help="候选池数量")
    parser.add_argument("--capital", type=float, default=500000.0, help="可用资金")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM Agent 层（仅量化选股）")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--model-path", type=str, default="data/xgb_scorer.model",
                        help="XGBoost 模型路径（不存在则等权打分）")
    return parser.parse_args()


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------


def run_daily(
    date: Optional[str] = None,
    top_n: int = 20,
    available_capital: float = 500000.0,
    skip_llm: bool = False,
    log_dir: str = "logs",
    model_path: str = "data/xgb_scorer.model",
) -> List[dict]:
    """
    执行单日完整交易流程。

    Parameters
    ----------
    date : str, optional
        YYYYMMDD。
    top_n : int
        候选池大小。
    available_capital : float
        可用资金。
    skip_llm : bool
        是否跳过 LLM 决策层。
    log_dir : str
        日志目录。

    Returns
    -------
    list[dict]
        赛制标准 JSON 格式的决策列表。
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    logger.info("=" * 60)
    logger.info("AlphaAgent 比赛流程启动 · 日期 %s", date)
    logger.info("=" * 60)

    # --------------- Layer 0~2: 量化选股 ---------------
    logger.info("[Layer 0~2] 量化选股开始...")
    screener = StockScreener(end_date=date)

    # 加载预训练 XGBoost 模型（不存在时自动降级为等权打分）
    screener.load_model(model_path)

    candidates = screener.run(top_n=top_n)
    logger.info("候选池: %d 只", len(candidates))

    if not candidates:
        logger.warning("候选池为空，当日空仓")
        return []

    # --------------- Layer 3: 多智能体决策 ---------------
    if not skip_llm:
        logger.info("[Layer 3] 多智能体决策开始...")
        candidates = _run_llm_layer(candidates)

    # 过滤 C_mixed <= 0.55 的标的
    candidates = [c for c in candidates if c.get("C_mixed", 0.5) > 0.55]
    logger.info("C_mixed > 0.55 的标的: %d 只", len(candidates))

    if not candidates:
        logger.warning("无符合条件的持仓标的，当日空仓")
        return []

    # 按 C_mixed 降序，取前 5
    candidates = sorted(candidates, key=lambda x: x.get("C_mixed", 0), reverse=True)
    candidates = candidates[:5]

    # --------------- Layer 4: 资金分配与输出 ---------------
    logger.info("[Layer 4] 资金分配与输出...")

    # 拉取前收盘价（从 stock_zh_a_hist 取最近一条）
    prev_closes = _fetch_prev_closes([c["symbol"] for c in candidates], date)

    sizer = PositionSizer()
    decisions = sizer.allocate(candidates, available_capital, prev_closes)

    formatter = OutputFormatter(log_dir=log_dir)
    json_output = formatter.format(decisions)
    formatter.write_log(decisions, date=date, available_capital=available_capital)

    logger.info("最终输出: %s", json_output)
    return json_output


# ------------------------------------------------------------------
# LLM 层（占位）
# ------------------------------------------------------------------


def _run_llm_layer(candidates: List[dict]) -> List[dict]:
    """
    多智能体决策层。
    当前为占位实现：用 C_mixed = xgb_score 作为模拟。

    TODO: 接入真实 Agent 并行调用框架。
    """
    for c in candidates:
        xgb = c.get("xgb_score", 0.5)
        # 模拟：C_raw 围绕 xgb_score 波动
        bull_confidence = min(1.0, max(0.1, xgb + 0.1))
        bear_confidence = min(1.0, max(0.1, 1.0 - xgb - 0.1))
        c_raw = (bull_confidence - bear_confidence + 1) / 2
        c_llm = xgb  # 模拟 LLM 打分
        c["bull_confidence"] = round(bull_confidence, 3)
        c["bear_confidence"] = round(bear_confidence, 3)
        c["C_llm"] = round(c_llm, 3)
        c["C_mixed"] = round(0.6 * c_raw + 0.4 * c_llm, 3)
    return candidates


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------


def _fetch_prev_closes(symbols: List[str], date: str) -> dict:
    """拉取前收盘价。"""
    from src.factors.price_volume import _fetch_one_stock

    result = {}
    for sym in symbols:
        try:
            df = _fetch_one_stock(sym, start_date=None, end_date=date)
            if df is not None and not df.empty and "close" in df.columns:
                result[sym] = float(df["close"].iloc[-1])
        except Exception:
            logger.exception("获取 %s 前收盘价失败", sym)
    return result


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    output = run_daily(
        date=args.date,
        top_n=args.top_n,
        available_capital=args.capital,
        skip_llm=args.skip_llm,
        log_dir=args.log_dir,
        model_path=args.model_path,
    )
    print("最终输出:", output)
    return output


if __name__ == "__main__":
    main()