"""
资金流因子模块。

使用 AKShare `stock_individual_fund_flow` 和 `stock_connect_flow` 提取
主力净流入、超大单净流入、北向资金等资金流因子。

对外接口:
    compute_capital_flow_factors(symbols, date) -> pd.DataFrame
"""

import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_capital_flow_factors(
    symbols: List[str],
    date: Optional[str] = None,
    request_interval: float = 0.3,
) -> pd.DataFrame:
    """
    批量计算资金流因子。

    Parameters
    ----------
    symbols : list[str]
        股票代码列表。
    date : str, optional
        目标日期 YYYYMMDD，默认最近交易日。
    request_interval : float
        请求间隔（秒）。

    Returns
    -------
    pd.DataFrame
        行索引 symbol，列:
        - main_net_ratio  : 主力净流入 / 总成交额
        - super_net_ratio : 超大单净流入 / 总成交额
        - main_net_5d     : 近 5 日累计主力净流入比
    """
    if not symbols:
        return pd.DataFrame()

    if date is None:
        date = pd.Timestamp.now().strftime("%Y%m%d")

    all_rows = []
    total = len(symbols)

    for idx, sym in enumerate(symbols):
        try:
            row = _fund_flow_one(sym, date)
            if row is not None:
                all_rows.append(row)
        except Exception:
            logger.exception("资金流因子计算 %s 失败", sym)

        if idx < total - 1:
            time.sleep(request_interval)

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    if "symbol" in result.columns:
        result.set_index("symbol", inplace=True)
    return result


def _fund_flow_one(symbol: str, date: str) -> Optional[dict]:
    """
    拉取单只股票近 5 日个股资金流向，计算因子。

    Returns
    -------
    dict or None
    """
    import akshare as ak

    try:
        df = ak.stock_individual_fund_flow(
            stock=symbol,
            market="sh" if symbol.startswith(("6", "9")) else "sz",
        )
    except Exception:
        logger.warning("拉取 %s 资金流失败", symbol)
        return None

    if df is None or df.empty:
        return None

    # 标准化列名
    col_map = {
        "日期": "date",
        "主力净流入-净额": "main_net",
        "超大单净流入-净额": "super_net",
        "成交额": "amount",
    }
    for old, new in col_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    required = ["main_net", "amount"]
    for c in required:
        if c not in df.columns:
            return None

    n = min(len(df), 5)
    if n == 0:
        return None

    recent = df.head(n)

    # 主力净流入比
    main_net = recent["main_net"].astype(float)
    amount = recent["amount"].astype(float).replace(0, np.nan)

    main_net_ratio = (main_net.iloc[0] / amount.iloc[0]) if not pd.isna(amount.iloc[0]) and amount.iloc[0] != 0 else np.nan
    main_net_5d = main_net.sum() / amount.sum() if amount.sum() != 0 else np.nan

    # 超大单
    super_net_ratio = np.nan
    if "super_net" in recent.columns:
        super_net = recent["super_net"].astype(float)
        super_net_ratio = (super_net.iloc[0] / amount.iloc[0]) if not pd.isna(amount.iloc[0]) and amount.iloc[0] != 0 else np.nan

    return {
        "symbol": symbol,
        "main_net_ratio": main_net_ratio,
        "super_net_ratio": super_net_ratio,
        "main_net_5d": main_net_5d,
    }