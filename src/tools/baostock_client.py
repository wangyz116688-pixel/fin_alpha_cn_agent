import atexit
import threading
from datetime import datetime
from typing import List

import baostock as bs
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("baostock_client")
_LOGIN_LOCK = threading.Lock()
_LOGGED_IN = False

FIELD_SET = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "pctChg",
]

ADJUST_FLAG_MAP = {
    "": "3",      # no adjustment
    "none": "3",
    "qfq": "2",   # pre-adjusted
    "hfq": "1",   # post-adjusted
}


def _logout() -> None:
    global _LOGGED_IN
    with _LOGIN_LOCK:
        if not _LOGGED_IN:
            return
        try:
            bs.logout()
            logger.info("BaoStock session closed.")
        except Exception:
            pass  # 网络中断时 logout 可能失败，忽略即可
        finally:
            _LOGGED_IN = False


def ensure_login() -> None:
    global _LOGGED_IN
    with _LOGIN_LOCK:
        if _LOGGED_IN:
            return
        rs = bs.login()
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {rs.error_msg}")
        _LOGGED_IN = True
        logger.info("BaoStock login successful.")
        atexit.register(_logout)


def format_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    lowered = symbol.lower()
    if lowered.startswith("sh.") or lowered.startswith("sz."):
        return lowered
    if symbol.startswith(("6", "9")):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


def _coerce_dates(value) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return str(value)


def query_history_k_data_plus(
    symbol: str,
    start_date,
    end_date,
    adjust: str = "qfq",
) -> pd.DataFrame:
    ensure_login()
    bs_symbol = format_symbol(symbol)
    start = _coerce_dates(start_date)
    end = _coerce_dates(end_date)
    adjust_flag = ADJUST_FLAG_MAP.get(adjust.lower() if adjust else "", "2")

    fields = ",".join(FIELD_SET)
    rs = bs.query_history_k_data_plus(
        bs_symbol,
        fields,
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag=adjust_flag,
    )
    if rs.error_code != "0":  # 登录失效或网络中断，重连后重试一次
        _logout()
        ensure_login()
        rs = bs.query_history_k_data_plus(
            bs_symbol,
            fields,
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag=adjust_flag,
        )
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock query failed[{rs.error_code}]: {rs.error_msg}")

    rows: List[List[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    df = pd.DataFrame(rows, columns=FIELD_SET)
    return df


def query_trade_dates(start_date, end_date) -> pd.DataFrame:
    ensure_login()
    start = _coerce_dates(start_date)
    end = _coerce_dates(end_date)
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    if rs.error_code != "0":  # 登录失效或网络中断，重连后重试一次
        _logout()
        ensure_login()
        rs = bs.query_trade_dates(start_date=start, end_date=end)
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock trade date query failed[{rs.error_code}]: {rs.error_msg}")
    rows: List[List[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=["calendar_date", "is_trading_day"])
    df = pd.DataFrame(rows, columns=rs.fields)
    return df


def query_stock_basic(code: str) -> pd.DataFrame:
    ensure_login()
    rs = bs.query_stock_basic(code=code)
    if rs.error_code != "0":  # 登录失效或网络中断，重连后重试一次
        _logout()
        ensure_login()
        rs = bs.query_stock_basic(code=code)
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock stock basic query failed[{rs.error_code}]: {rs.error_msg}")

    rows: List[List[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=rs.fields)
    df = pd.DataFrame(rows, columns=rs.fields)
    return df
