"""
输出格式化模块。

生成赛制要求的 JSON 输出，同时写入审计日志 reasoning.md。

对外接口:
    OutputFormatter 类
        - format(decisions) -> list[dict]  (标准 JSON)
        - write_log(decisions, date) -> str  (写 reasoning.md)
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    输出格式化器。

    Parameters
    ----------
    log_dir : str
        审计日志目录，默认 "logs"。
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 标准 JSON 输出
    # ------------------------------------------------------------------

    def format(self, decisions: List[dict]) -> List[dict]:
        """
        将决策列表转换为赛制标准 JSON 格式。

        Parameters
        ----------
        decisions : list[dict]
            每项需含: symbol, symbol_name, volume

        Returns
        -------
        list[dict]
            [{"symbol": "600519", "symbol_name": "贵州茅台", "volume": 100}, ...]
        """
        result = []
        for d in decisions:
            volume = d.get("volume", 0)
            if volume <= 0:
                continue
            result.append(
                {
                    "symbol": str(d["symbol"]),
                    "symbol_name": str(d.get("symbol_name", "")),
                    "volume": int(volume),
                }
            )
        return result

    # ------------------------------------------------------------------
    # 审计日志
    # ------------------------------------------------------------------

    def write_log(
        self,
        decisions: List[dict],
        date: Optional[str] = None,
        available_capital: float = 500000.0,
    ) -> str:
        """
        写入 reasoning.md 审计日志。

        Parameters
        ----------
        decisions : list[dict]
            完整决策信息（含 C_mixed, key_factors 等）。
        date : str, optional
            日期 YYYYMMDD，默认今天。
        available_capital : float
            当日可用资金。

        Returns
        -------
        str
            日志文件路径。
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        json_path = os.path.join(self.log_dir, f"{date}_decision.json")
        md_path = os.path.join(self.log_dir, f"{date}_reasoning.md")

        # 写 JSON 决策文件
        json_output = self.format(decisions)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        logger.info("决策 JSON 已写入 %s", json_path)

        # 写 reasoning.md
        lines = []
        lines.append(f"# {date} 交易决策审计日志")
        lines.append("")
        lines.append(f"> 可用资金总额: {available_capital:,.0f} 元")
        lines.append(f"> 持仓标的数: {len(json_output)}")
        lines.append("")

        if not decisions:
            lines.append("## 当日空仓")
            lines.append("")
            lines.append("所有候选标的 C_mixed ≤ 0.55 或无候选池。")
        else:
            for i, d in enumerate(decisions, 1):
                sym = d.get("symbol", "?")
                name = d.get("symbol_name", "")
                c_mixed = d.get("C_mixed", 0)
                xgb = d.get("xgb_score", 0)
                volume = d.get("volume", 0)
                weight = d.get("weight", 0)
                amount = d.get("amount", 0)
                prev_close = d.get("prev_close", 0)
                industry = d.get("industry", "未知")
                cap = d.get("market_cap_bn", "")

                lines.append(f"## {i}. {sym} {name} | C_mixed: {c_mixed:.3f}")
                lines.append("")
                lines.append(f"- **XGB 综合得分**: {xgb:.4f}")
                lines.append(f"- **行业**: {industry}")
                lines.append(f"- **市值**: {cap} 亿元")
                lines.append("")

                # 关键因子
                key_factors = d.get("key_factors", {})
                if key_factors:
                    lines.append("### 关键因子")
                    for k, v in list(key_factors.items())[:6]:
                        lines.append(f"- {k}: {v}")
                    lines.append("")

                # 资金分配
                lines.append("### 资金分配")
                lines.append(f"- 仓位权重: {weight:.2%}")
                lines.append(f"- 买入金额: {amount:,.0f} 元")
                lines.append(f"- 昨日收盘: {prev_close:.2f} 元")
                lines.append(f"- 建议买入: {volume} 股")
                lines.append("")

                # 多空辩论（若有）
                bull = d.get("bull_confidence")
                bear = d.get("bear_confidence")
                c_llm = d.get("C_llm")
                if bull is not None and bear is not None:
                    lines.append("### 多空辩论结论")
                    lines.append(f"- 多方置信度: {bull:.2f}")
                    lines.append(f"- 空方置信度: {bear:.2f}")
                    if c_llm is not None:
                        lines.append(f"- LLM 第三方评分: {c_llm:.2f}")
                    lines.append(f"- **C_mixed** = 0.6 × C_raw + 0.4 × C_llm = **{c_mixed:.3f}**")
                    lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("审计日志已写入 %s", md_path)

        return md_path