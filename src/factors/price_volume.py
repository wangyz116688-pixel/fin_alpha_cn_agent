"""
量价因子计算模块。

使用 AKShare `stock_zh_a_hist` 拉取日频前复权 K 线，
批量计算动量、波动率、换手率、量价关系等因子，返回 DataFrame。

对外接口:
    compute_price_volume_factors(symbols, start_date, end_date, lookback_days) -> pd.DataFrame
"""

import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def compute_price_volume_factors(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: int = 60,
    request_interval: float = 0.3,
) -> pd.DataFrame:
    """
    批量计算量价因子，返回 DataFrame。

    Parameters
    ----------
    symbols : list[str]
        股票代码列表，格式如 ["000001", "600519"]（不含市场后缀）。
    start_date : str, optional
        数据起始日期，格式 "YYYYMMDD"。默认取 end_date 向前推 lookback_days。
    end_date : str, optional
        数据截止日期，格式 "YYYYMMDD"。默认今天。
    lookback_days : int
        回溯天数（用于计算因子需要的窗口长度）。
    request_interval : float
        每次 AKShare 请求间隔秒数，避免触发限频。

    Returns
    -------
    pd.DataFrame
        行索引为 symbol，列为各因子名。
        因子列：
        - mom_1d, mom_5d, mom_20d, rev_1d
        - vol_5d, vol_20d, vol_ratio
        - turn_1d, turn_5d_avg, turn_ratio
        - price_vol_corr_5, price_turn_corr_5, vwap_dev
    """
    if not symbols:
        return pd.DataFrame()

    # 处理日期默认值
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y%m%d")
    if start_date is None:
        # 往前多取一些以保证计算窗口足够（20 日动量需要 21 天，再加缓冲）
        start_date = (
            pd.Timestamp(end_date) - pd.Timedelta(days=max(lookback_days + 30, 90))
        ).strftime("%Y%m%d")

    all_rows = []
    total = len(symbols)

    for idx, symbol in enumerate(symbols):
        try:
            raw_df = _fetch_one_stock(symbol, start_date, end_date)
            if raw_df is None or raw_df.empty:
                logger.warning("跳过 %s：无数据", symbol)
                continue

            factor_series = _compute_factors(raw_df, symbol)
            if factor_series is not None:
                all_rows.append(factor_series)
        except Exception:
            logger.exception("计算 %s 因子时出错，跳过", symbol)

        # 限频
        if idx < total - 1:
            time.sleep(request_interval)

    if not all_rows:
        logger.warning("没有成功计算出任何股票的因子")
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    result.set_index("symbol", inplace=True)
    return result


# ---------------------------------------------------------------------------
# 内部函数
# ---------------------------------------------------------------------------


def _fetch_one_stock(
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    拉取单只股票日频前复权 K 线。

    优先使用 baostock，失败时回退到 AKShare。

    Parameters
    ----------
    symbol : str
        股票代码，如 "000001"。
    start_date : str or None
        起始日期 "YYYYMMDD"。
    end_date : str or None
        截止日期 "YYYYMMDD"。

    Returns
    -------
    pd.DataFrame or None
    """
    df = _fetch_one_stock_baostock(symbol, start_date, end_date)
    if df is not None and not df.empty:
        return df
    logger.warning("baostock 拉取 %s 失败，回退到 AKShare", symbol)
    return _fetch_one_stock_akshare(symbol, start_date, end_date)


def _fmt_date_bs(date_str: Optional[str]) -> Optional[str]:
    """将 YYYYMMDD 转为 YYYY-MM-DD（baostock 需要）。"""
    if date_str is None:
        return None
    s = str(date_str).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _fetch_one_stock_baostock(
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    from src.tools.baostock_client import query_history_k_data_plus

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=120)).strftime("%Y%m%d")

    try:
        raw = query_history_k_data_plus(
            symbol=symbol,
            start_date=_fmt_date_bs(start_date),
            end_date=_fmt_date_bs(end_date),
            adjust="qfq",
        )
    except Exception:
        logger.exception("baostock 拉取 %s 失败", symbol)
        return None

    if raw is None or raw.empty:
        return None

    # baostock 字段: date, code, open, high, low, close, preclose, volume, amount, turn, pctChg
    col_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
        "turn": "turn",
        "pctChg": "pct_change",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pct_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_one_stock_akshare(
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    import akshare as ak
    from src.network.proxy_manager import proxy_manager

    try:
        df = proxy_manager.run(
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            ),
            f"stock_zh_a_hist({symbol})",
        )
    except Exception:
        logger.exception("AKShare 拉取 %s 失败", symbol)
        return None

    if df is None or df.empty:
        return None

    _normalize_columns(df)
    return df


def _normalize_columns(df: pd.DataFrame) -> None:
    """将 AKShare 中文列名映射为英文缩写，原地修改。"""
    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turn",
    }
    for old, new in col_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])


def _compute_factors(
    df: pd.DataFrame,
    symbol: str,
) -> Optional[pd.Series]:
    """
    对单只股票的原始 K 线 DataFrame 计算全部量价因子，
    返回最新一天（T-1）的因子 Series。

    Parameters
    ----------
    df : pd.DataFrame
        标准化后的 K 线数据（按日期升序排列，最早日期排在最前）。
    symbol : str
        股票代码。

    Returns
    -------
    pd.Series or None
        若数据不足计算因子则返回 None。
    """
    # 确保按日期排序（升序）
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    required_cols = ["close", "volume", "amount", "turn"]
    for col in required_cols:
        if col not in df.columns:
            logger.warning("%s 缺少列 %s，无法计算因子", symbol, col)
            return None

    n = len(df)
    # 需要至少 22 行才能计算 20 日动量（close[t-1] / close[t-21]）
    if n < 22:
        logger.warning("%s 数据不足（%d 行），跳过", symbol, n)
        return None

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)
    turn = df["turn"].astype(float)

    # 日收益率
    daily_ret = close.pct_change()

    # ---- 动量因子 ----
    # 使用 t-1（最新一天）的 close
    close_now = close.iloc[-1]  # t-1

    mom_1d = np.nan
    if n >= 3:
        mom_1d = close_now / close.iloc[-2] - 1.0  # close[t-1] / close[t-2] - 1

    mom_5d = np.nan
    if n >= 7:
        mom_5d = close_now / close.iloc[-6] - 1.0  # close[t-1] / close[t-6] - 1

    mom_20d = np.nan
    if n >= 22:
        mom_20d = close_now / close.iloc[-21] - 1.0  # close[t-1] / close[t-21] - 1

    rev_1d = -mom_1d if not np.isnan(mom_1d) else np.nan

    # ---- 波动率因子 ----
    vol_5d = daily_ret.rolling(5).std().iloc[-1] if len(daily_ret.dropna()) >= 5 else np.nan
    vol_20d = daily_ret.rolling(20).std().iloc[-1] if len(daily_ret.dropna()) >= 20 else np.nan
    vol_ratio = vol_5d / vol_20d if (vol_5d is not None and vol_20d is not None and not np.isnan(vol_5d) and not np.isnan(vol_20d) and vol_20d != 0) else np.nan

    # ---- 换手率因子 ----
    turn_1d = turn.iloc[-1] if not np.isnan(turn.iloc[-1]) else np.nan
    turn_5d_avg = turn.rolling(5).mean().iloc[-1] if n >= 5 else np.nan
    turn_ratio = turn_1d / turn_5d_avg if (not np.isnan(turn_1d) and not np.isnan(turn_5d_avg) and turn_5d_avg != 0) else np.nan

    # ---- 量价关系因子 ----
    price_vol_corr_5 = close.rolling(5).corr(volume).iloc[-1] if n >= 5 else np.nan
    price_turn_corr_5 = close.rolling(5).corr(turn).iloc[-1] if n >= 5 else np.nan

    # VWAP 偏离: (close - vwap) / vwap
    # AKShare 成交量单位为"手"（100 股），需换算为股数
    vwap_dev = np.nan
    if n >= 1:
        vol_shares = volume.iloc[-1] * 100  # 手 -> 股
        vwap = amount.iloc[-1] / vol_shares if vol_shares != 0 else np.nan
        if not np.isnan(vwap) and vwap != 0:
            vwap_dev = (close_now - vwap) / vwap

    # 构建结果 Series
    series = pd.Series(
        {
            "symbol": symbol,
            "mom_1d": mom_1d,
            "mom_5d": mom_5d,
            "mom_20d": mom_20d,
            "rev_1d": rev_1d,
            "vol_5d": vol_5d,
            "vol_20d": vol_20d,
            "vol_ratio": vol_ratio,
            "turn_1d": turn_1d,
            "turn_5d_avg": turn_5d_avg,
            "turn_ratio": turn_ratio,
            "price_vol_corr_5": price_vol_corr_5,
            "price_turn_corr_5": price_turn_corr_5,
            "vwap_dev": vwap_dev,
        }
    )
    return series