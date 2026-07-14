"""
竞赛口径多股票量化回测。

结算规则：
  - 买入价：前一交易日收盘价
  - 卖出价：当日收盘价（日终自动卖出，不留隔夜）
  - 单笔盈亏 = amount × (当日收盘 - 昨收) / 昨收
  - 次日可用资金 = 累计总资产

使用方式（MVP 快速测试）：
    python -m src.backtest_stock_agent --mvp

使用方式（完整回测）：
    python -m src.backtest_stock_agent --start 20260518 --end 20260613 --capital 500000 --top-n 20

断点续传：程序中断后直接重新运行同样命令，自动从断点继续。
缓存目录默认为 data/prefetch_cache/，可用 --cache-dir 指定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.stock_screener import StockScreener
from src.execution.position_sizer import PositionSizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
FACTOR_CACHE_VERSION = "v2"


# ---------------------------------------------------------------------------
# 断点续传：单股缓存
# ---------------------------------------------------------------------------

def _stock_cache_path(cache_dir: str, sym: str, start_date: str, end_date: str) -> str:
    return os.path.join(cache_dir, f"{sym}_{start_date}_{end_date}.pkl")


def _parse_stock_cache_name(filename: str) -> Optional[Tuple[str, str, str]]:
    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".pkl":
        return None
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    sym, start, end = parts
    if not (sym.isdigit() and len(start) == 8 and len(end) == 8):
        return None
    return sym, start, end


def _cached_symbols_covering(cache_dir: str, start_date: str, end_date: str) -> set[str]:
    """Return symbols whose cached files cover the requested date span."""
    if not os.path.isdir(cache_dir):
        return set()

    spans: Dict[str, List[Tuple[str, str]]] = {}
    for name in os.listdir(cache_dir):
        parsed = _parse_stock_cache_name(name)
        if parsed is None:
            continue
        sym, cached_start, cached_end = parsed
        if cached_end < start_date or cached_start > end_date:
            continue
        spans.setdefault(sym, []).append((cached_start, cached_end))

    covered = set()
    for sym, ranges in spans.items():
        min_start = min(start for start, _ in ranges)
        max_end = max(end for _, end in ranges)
        if min_start <= start_date and max_end >= end_date:
            covered.add(sym)
    return covered


def _load_cached_stock(cache_dir: str, sym: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    path = _stock_cache_path(cache_dir, sym, start_date, end_date)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    if not os.path.isdir(cache_dir):
        return None

    frames = []
    ranges = []
    for name in os.listdir(cache_dir):
        parsed = _parse_stock_cache_name(name)
        if parsed is None:
            continue
        cached_sym, cached_start, cached_end = parsed
        if cached_sym != sym:
            continue
        if cached_end < start_date or cached_start > end_date:
            continue
        ranges.append((cached_start, cached_end))
        try:
            with open(os.path.join(cache_dir, name), "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return None
    if min(start for start, _ in ranges) > start_date or max(end for _, end in ranges) < end_date:
        return None

    try:
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        start_ts = pd.Timestamp(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}")
        end_ts = pd.Timestamp(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _save_cached_stock(cache_dir: str, sym: str, start_date: str, end_date: str, df: pd.DataFrame) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _stock_cache_path(cache_dir, sym, start_date, end_date)
    try:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass


def _stock_info_from_price_cache(cache_dir: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Build a minimal stock universe from cached K-line files for offline runs."""
    if not os.path.isdir(cache_dir):
        return pd.DataFrame()

    symbols = set()
    for name in os.listdir(cache_dir):
        parsed = _parse_stock_cache_name(name)
        if parsed is None:
            continue
        sym, cached_start, cached_end = parsed
        if cached_end < start_date or cached_start > end_date:
            continue
        symbols.add(sym)
    symbols = sorted(symbols)
    if not symbols:
        return pd.DataFrame()

    df = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
    df["symbol_name"] = symbols
    df["industry"] = "未知"
    df["pe_ttm"] = np.nan
    df["market_cap_bn"] = np.nan
    df["is_st"] = False
    df["chg_60d"] = np.nan
    logger.info("离线模式: 从 K 线缓存恢复股票池 %d 只", len(df))
    return df


def _factor_cache_path(
    factor_cache_dir: str,
    symbols: List[str],
    as_of_date: pd.Timestamp,
    lookback_days: int,
    neutralized: bool,
) -> str:
    universe = "|".join(sorted(symbols))
    universe_hash = hashlib.sha1(universe.encode("utf-8")).hexdigest()[:12]
    tag = "neutral" if neutralized else "raw"
    filename = (
        f"factors_{FACTOR_CACHE_VERSION}_{as_of_date.strftime('%Y%m%d')}"
        f"_lb{lookback_days}_{tag}_{len(symbols)}_{universe_hash}.pkl"
    )
    return os.path.join(factor_cache_dir, filename)


# ---------------------------------------------------------------------------
# 数据预拉取（支持断点续传）
# ---------------------------------------------------------------------------

def prefetch_price_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    request_interval: float = 0.2,
    cache_dir: str = "data/prefetch_cache",
    max_retries: int = 5,
    offline: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    批量拉取全市场 K 线，支持断点续传。

    - 每只股票拉完后立即写盘（pkl），中断重启自动跳过已缓存股票
    - 网络失败自动重试（指数退避，最多 max_retries 次）
    - offline=True 时完全不联网：只用本地缓存，未缓存的股票直接跳过
    """
    from src.tools.baostock_client import query_history_k_data_plus

    def fmt(d: str) -> str:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"

    s_date = fmt(start_date)
    e_date = fmt(end_date)

    price_map: Dict[str, pd.DataFrame] = {}
    total = len(symbols)
    skipped = 0
    fetched = 0
    failed = 0

    # 先加载已有缓存
    for sym in symbols:
        cached = _load_cached_stock(cache_dir, sym, start_date, end_date)
        if cached is not None:
            price_map[sym] = cached
            skipped += 1

    remaining = [s for s in symbols if s not in price_map]
    logger.info(
        "预拉取计划: 共 %d 只 | 已缓存 %d 只（跳过）| 待拉取 %d 只",
        total, skipped, len(remaining),
    )
    if not remaining:
        logger.info("全部来自缓存，无需网络请求。")
        return price_map

    if offline:
        logger.info(
            "离线模式：跳过 %d 只未缓存股票，仅用 %d 只缓存数据回测。",
            len(remaining), skipped,
        )
        return price_map

    logger.info("开始拉取剩余 %d 只（%s ~ %s）...", len(remaining), s_date, e_date)

    for i, sym in enumerate(remaining):
        if (i + 1) % 100 == 0:
            logger.info("  进度: %d / %d（本次新拉取）| 已成功 %d | 失败 %d",
                        i + 1, len(remaining), fetched, failed)

        df = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = query_history_k_data_plus(
                    symbol=sym,
                    start_date=s_date,
                    end_date=e_date,
                    adjust="qfq",
                )
                if raw is None or raw.empty:
                    break
                col_map = {
                    "date": "date", "open": "open", "high": "high",
                    "low": "low", "close": "close", "volume": "volume",
                    "amount": "amount", "turn": "turn", "pctChg": "pct_change",
                }
                df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
                df["date"] = pd.to_datetime(df["date"])
                for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pct_change"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                break
            except Exception as e:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "%s 拉取失败（第 %d/%d 次）: %s | 等待 %ds 重试",
                    sym, attempt, max_retries, e, wait,
                )
                time.sleep(wait)

        if df is not None and not df.empty:
            price_map[sym] = df
            _save_cached_stock(cache_dir, sym, start_date, end_date, df)
            fetched += 1
        else:
            failed += 1

        if i < len(remaining) - 1:
            time.sleep(request_interval)

    logger.info(
        "预拉取完成: 总计 %d 只 | 缓存命中 %d | 新拉取成功 %d | 失败 %d",
        total, skipped, fetched, failed,
    )
    return price_map


# ---------------------------------------------------------------------------
# 因子计算（基于缓存数据）
# ---------------------------------------------------------------------------

def compute_cross_section(
    price_map: Dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    screener: Optional["StockScreener"] = None,
    lookback_days: int = 90,
    factor_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    基于缓存 K 线，计算截至 as_of_date（含）的完整截面因子矩阵：
    量价因子(13) + AlphaNet 特征(10)，并（传入 screener 时）做五因子中性化。

    与训练端 [train_xgboost.py] 和推理端 [StockScreener.run] 保持因子一致，
    保证"训练 → 回测/选股"特征对齐。
    """
    from src.factors.price_volume import _compute_factors
    from src.factors.alphanet_features import compute_alphanet_features

    cutoff = as_of_date
    start_window = cutoff - pd.Timedelta(days=lookback_days + 30)
    cache_path = None
    if factor_cache_dir:
        cache_path = _factor_cache_path(
            factor_cache_dir,
            list(price_map.keys()),
            as_of_date,
            lookback_days,
            neutralized=screener is not None,
        )
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    factor_df = pickle.load(f)
                if isinstance(factor_df, pd.DataFrame):
                    logger.info("因子缓存命中: %s", cache_path)
                    return factor_df
            except Exception:
                logger.warning("因子缓存读取失败，将重新计算: %s", cache_path)

    pv_rows = []
    alpha_rows = []
    for sym, df in price_map.items():
        if df[df["date"] == cutoff].empty:
            continue
        window = df[(df["date"] >= start_window) & (df["date"] <= cutoff)].copy()
        if window.empty or len(window) < 22:
            continue
        try:
            pv = _compute_factors(window, sym)
            if pv is not None:
                pv_rows.append(pv)
            al = compute_alphanet_features(window, sym)
            if al is not None:
                alpha_rows.append(al)
        except Exception:
            pass

    if not pv_rows:
        return pd.DataFrame()

    pv_df = pd.DataFrame(pv_rows).set_index("symbol")
    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows).set_index("symbol")
        factor_df = pv_df.join(alpha_df, how="inner")
    else:
        factor_df = pv_df

    # 五因子中性化（baostock 无行业/市值时自动降级为仅风格中性化）
    if screener is not None:
        empty_info = pd.DataFrame(index=factor_df.index)
        factor_df = screener._neutralize(factor_df, empty_info)

    if cache_path:
        try:
            os.makedirs(factor_cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(factor_df, f)
            logger.info("因子缓存已保存: %s", cache_path)
        except Exception:
            logger.warning("因子缓存写入失败: %s", cache_path)
    return factor_df


# ---------------------------------------------------------------------------
# 每日筛股
# ---------------------------------------------------------------------------

def screen_for_date(
    screener: StockScreener,
    price_map: Dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    stock_info: pd.DataFrame,
    top_n: int = 20,
    factor_cache_dir: Optional[str] = None,
    score_method: str = "ensemble_blend",
) -> List[dict]:
    factor_df = compute_cross_section(
        price_map,
        as_of_date,
        screener,
        factor_cache_dir=factor_cache_dir,
    )
    if factor_df.empty:
        return []

    valid_syms = sorted(set(stock_info.index) & set(factor_df.index))
    factor_df = factor_df.loc[valid_syms]
    if factor_df.empty:
        return []

    factor_df = screener._score(factor_df)
    factor_df = _apply_score_method(factor_df, score_method)
    candidates = screener._build_candidates(factor_df, stock_info, top_n)

    for c in candidates:
        if "C_mixed" not in c:
            c["C_mixed"] = c.get("xgb_score", 0.5)

    return candidates


def _rank_pct(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    return clean.rank(method="average", pct=True).fillna(0.5)


def _apply_score_method(factor_df: pd.DataFrame, score_method: str) -> pd.DataFrame:
    """Optionally replace xgb_score with an IC-informed blended rank score."""
    method = (score_method or "xgb").lower()
    if method == "xgb":
        return factor_df
    if method == "ic_blend":
        blend_cols = ["mom_20d", "decay_close_5", "std_close_10"]
        weights = {col: 1.0 for col in blend_cols}
    elif method == "recent_ic_blend":
        weights = {
            "std_close_10": 0.040301,
            "decay_close_5": 0.036596,
            "rev_1d": 0.011555,
            "ret_close_10": 0.006174,
        }
    elif method == "ensemble_blend":
        ic_cols = ["mom_20d", "decay_close_5", "std_close_10"]
        available = [c for c in ic_cols if c in factor_df.columns]
        if not available:
            logger.warning("ensemble_blend has no IC factors; falling back to xgb_score")
            return factor_df

        factor_df = factor_df.copy()
        factor_df["model_score"] = factor_df.get("xgb_score", 0.5)
        model_rank = _rank_pct(factor_df["model_score"])
        ic_rank = pd.concat([_rank_pct(factor_df[col]) for col in available], axis=1).mean(axis=1)
        factor_df["xgb_score"] = (0.5 * model_rank + 0.5 * ic_rank).clip(0.0, 1.0)
        return factor_df
    elif method == "train_ic_blend":
        weights = {
            "turn_5d_avg": -0.055547,
            "mom_20d": -0.052257,
            "vol_20d": -0.049372,
            "vwap_dev": -0.034496,
            "zscore_vol_5": -0.030165,
            "turn_ratio": -0.027209,
            "rev_1d": 0.014613,
            "std_vol_10": 0.007617,
            "std_close_10": 0.005100,
            "decay_vol_5": 0.004183,
        }
    else:
        raise ValueError(f"Unsupported score_method: {score_method}")

    available = [c for c in weights if c in factor_df.columns]
    if not available:
        logger.warning("%s has no available factors; falling back to xgb_score", score_method)
        return factor_df

    ranks = []
    rank_weights = []
    for col in available:
        weight = weights[col]
        ranks.append(_rank_pct(factor_df[col] if weight > 0 else -factor_df[col]))
        rank_weights.append(abs(weight))

    factor_df = factor_df.copy()
    factor_df["model_score"] = factor_df.get("xgb_score", 0.5)
    rank_df = pd.concat(ranks, axis=1)
    factor_df["xgb_score"] = np.average(rank_df.values, axis=1, weights=np.array(rank_weights))
    factor_df["xgb_score"] = pd.Series(factor_df["xgb_score"], index=factor_df.index).clip(0.0, 1.0)
    return factor_df


def _select_target_positions(
    candidates: List[dict],
    position_count: int,
    confidence_threshold: Optional[float],
    fill_positions: bool,
) -> List[dict]:
    if not candidates or position_count <= 0:
        return []

    ordered = sorted(candidates, key=lambda x: x.get("C_mixed", 0), reverse=True)
    if confidence_threshold is None:
        return ordered[:position_count]

    high_conf = [c for c in ordered if c.get("C_mixed", 0) > confidence_threshold]
    selected = high_conf[:position_count]
    if fill_positions and len(selected) < position_count:
        selected_symbols = {c["symbol"] for c in selected}
        for c in ordered:
            if c["symbol"] in selected_symbols:
                continue
            selected.append(c)
            selected_symbols.add(c["symbol"])
            if len(selected) >= position_count:
                break

    if selected:
        return selected
    return ordered[:position_count]


def _candidate_prev_closes(
    candidates: List[dict],
    price_map: Dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
) -> Dict[str, float]:
    prev_closes: Dict[str, float] = {}
    for candidate in candidates:
        sym = candidate["symbol"]
        if sym not in price_map:
            continue
        rows = price_map[sym][price_map[sym]["date"] == as_of_date]
        if rows.empty:
            continue
        prev_close = float(rows.iloc[-1]["close"])
        if prev_close > 0 and np.isfinite(prev_close):
            prev_closes[sym] = prev_close
    return prev_closes


def _filter_buyable_candidates(
    candidates: List[dict],
    prev_closes: Dict[str, float],
    available_capital: float,
    position_count: int,
    sizer: PositionSizer,
    allocation_method: str,
    max_stock_price: Optional[float] = None,
) -> List[dict]:
    if not candidates:
        return []

    if (allocation_method or "score").lower() == "equal":
        amount_limit = available_capital * (1.0 - sizer.buffer_ratio) / max(position_count, 1)
    else:
        amount_limit = available_capital * sizer.max_single_ratio
    amount_limit = min(amount_limit, available_capital * sizer.max_single_ratio)

    buyable = []
    for candidate in candidates:
        sym = candidate["symbol"]
        prev_close = prev_closes.get(sym)
        if prev_close is None:
            continue
        if max_stock_price is not None and prev_close > max_stock_price:
            continue
        if prev_close * sizer.min_volume > amount_limit:
            continue
        buyable.append(candidate)
    return buyable


def _build_decisions_for_method(
    screener: StockScreener,
    price_map: Dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    stock_info: pd.DataFrame,
    top_n: int,
    factor_cache_dir: Optional[str],
    score_method: str,
    available_capital: float,
    position_count: int,
    confidence_threshold: Optional[float],
    fill_positions: bool,
    allocation_method: str,
    sizer: PositionSizer,
    max_stock_price: Optional[float] = None,
) -> List[dict]:
    candidates = screen_for_date(
        screener,
        price_map,
        as_of_date,
        stock_info,
        top_n,
        factor_cache_dir=factor_cache_dir,
        score_method=score_method,
    )
    if not candidates:
        return []

    prev_closes = _candidate_prev_closes(candidates, price_map, as_of_date)
    candidates = _filter_buyable_candidates(
        candidates,
        prev_closes=prev_closes,
        available_capital=available_capital,
        position_count=position_count,
        sizer=sizer,
        allocation_method=allocation_method,
        max_stock_price=max_stock_price,
    )
    if not candidates:
        return []

    selected = _select_target_positions(
        candidates,
        position_count=position_count,
        confidence_threshold=confidence_threshold,
        fill_positions=fill_positions,
    )
    decisions = [dict(item) for item in selected]
    decisions = sizer.allocate(decisions, available_capital, prev_closes)
    return [item for item in decisions if item.get("volume", 0) > 0]


def _select_adaptive_method(
    history: Dict[str, List[float]],
    lookback: int,
    min_observations: int,
    cash_threshold: float,
    default_method: str,
    switch_margin: float,
) -> str:
    scores = {}
    for method, returns in history.items():
        recent = returns[-lookback:] if lookback > 0 else returns
        if len(recent) < min_observations:
            continue
        scores[method] = float(np.mean(recent))

    if not scores:
        return default_method

    best_method, best_score = max(scores.items(), key=lambda item: item[1])
    default_score = scores.get(default_method)

    if best_score <= cash_threshold:
        return "cash"
    if default_score is None:
        return best_method
    if default_score <= cash_threshold and best_score > cash_threshold:
        return best_method
    if best_method != default_method and best_score >= default_score + switch_margin:
        return best_method
    return default_method


# ---------------------------------------------------------------------------
# 结算
# ---------------------------------------------------------------------------

def settle_day(
    decisions: List[dict],
    price_map: Dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    available_capital: float,
) -> Tuple[float, List[dict], pd.DataFrame]:
    records = []
    total_pnl = 0.0

    for d in decisions:
        sym = d["symbol"]
        prev_close = d.get("prev_close", 0.0)
        volume = d.get("volume", 0)
        amount = d.get("amount", 0.0)

        if volume == 0 or prev_close <= 0:
            continue

        today_close = None
        if sym in price_map:
            rows = price_map[sym][price_map[sym]["date"] == trade_date]
            if not rows.empty:
                today_close = float(rows.iloc[-1]["close"])

        if today_close is None or today_close <= 0:
            today_close = prev_close

        pnl = amount * (today_close - prev_close) / prev_close
        total_pnl += pnl

        records.append({
            "symbol": sym,
            "symbol_name": d.get("symbol_name", ""),
            "prev_close": round(prev_close, 4),
            "today_close": round(today_close, 4),
            "volume": volume,
            "amount": round(amount, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round((today_close - prev_close) / prev_close * 100, 4),
        })

    detail_df = pd.DataFrame(records) if records else pd.DataFrame()
    next_capital = available_capital + total_pnl
    return next_capital, records, detail_df


# ---------------------------------------------------------------------------
# 主回测循环
# ---------------------------------------------------------------------------

def run_backtest(
    start_date: str,
    end_date: str,
    initial_capital: float = 500_000.0,
    top_n: int = 20,
    max_stocks: Optional[int] = None,
    request_interval: float = 0.2,
    cache_dir: str = "data/prefetch_cache",
    log_dir: str = "logs",
    model_path: str = "data/xgb_scorer.model",
    offline: bool = False,
    factor_cache_dir: Optional[str] = None,
    no_plot: bool = False,
    score_method: str = "train_ic_blend",
    position_count: int = 3,
    confidence_threshold: Optional[float] = 0.55,
    fill_positions: bool = True,
    allocation_method: str = "score",
    print_advice: bool = True,
    adaptive_methods: Optional[List[str]] = None,
    adaptive_lookback: int = 5,
    adaptive_min_observations: int = 3,
    adaptive_cash_threshold: float = -0.002,
    adaptive_default_method: str = "recent_ic_blend",
    adaptive_switch_margin: float = 0.003,
    max_drawdown_stop: Optional[float] = 0,
    risk_cooldown_days: int = 10,
    max_stock_price: Optional[float] = None,
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("竞赛回测  %s ~ %s  资金 %.0f  股票上限 %s",
                start_date, end_date, initial_capital,
                str(max_stocks) if max_stocks else "全市场")
    logger.info("=" * 60)

    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=120)).strftime("%Y%m%d")

    # ---- Step 1: 股票列表 ----
    screener = StockScreener(end_date=end_date, request_interval=request_interval)
    screener.load_model(model_path)   # 不存在时自动降级为等权打分
    stock_info = screener.get_stock_list(offline=offline)
    if stock_info.empty and offline:
        stock_info = _stock_info_from_price_cache(cache_dir, fetch_start, end_date)
    if stock_info.empty:
        logger.error("获取股票列表失败")
        return pd.DataFrame()

    tradable_info = screener.filter_tradable(stock_info)
    symbols = sorted(tradable_info.index.tolist())
    if offline:
        cached_symbols = _cached_symbols_covering(cache_dir, fetch_start, end_date)
        if cached_symbols:
            symbols = sorted(set(symbols) & cached_symbols)
            tradable_info = tradable_info.loc[tradable_info.index.isin(symbols)]
            logger.info("Offline cache-covered universe: %d symbols", len(symbols))
        else:
            logger.warning("Offline mode found no cache-covered symbols for %s ~ %s", fetch_start, end_date)

    if max_stocks and max_stocks < len(symbols):
        # 随机采样以覆盖不同行业
        import random
        random.seed(42)
        symbols = random.sample(symbols, max_stocks)
        tradable_info = tradable_info.loc[tradable_info.index.isin(symbols)]
        logger.info("MVP 模式：从全市场随机抽取 %d 只股票", max_stocks)
    else:
        logger.info("可交易标的: %d 只", len(symbols))

    # ---- Step 2: 预拉取（断点续传）----
    logger.info("[Step 2] 预拉取 K 线（缓存目录: %s）", cache_dir)
    price_map = prefetch_price_data(
        symbols, fetch_start, end_date,
        request_interval=request_interval,
        cache_dir=cache_dir,
        offline=offline,
    )
    if not price_map:
        logger.error("K 线数据拉取失败")
        return pd.DataFrame()

    # ---- Step 3: 确定交易日序列 ----
    all_dates: set = set()
    for df in price_map.values():
        all_dates.update(df["date"].tolist())

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    trade_dates = sorted(d for d in all_dates if start_ts <= d <= end_ts)

    if not trade_dates:
        logger.error("回测区间内无交易日")
        return pd.DataFrame()

    logger.info("[Step 3] 交易日: %d 个 (%s ~ %s)",
                len(trade_dates),
                trade_dates[0].strftime("%Y-%m-%d"),
                trade_dates[-1].strftime("%Y-%m-%d"))

    # ---- Step 4: 逐日回测 ----
    from src.execution.output_formatter import OutputFormatter

    sizer = PositionSizer(allocation_method=allocation_method)
    formatter = OutputFormatter(log_dir=log_dir)
    capital = initial_capital
    daily_records = []
    daily_advice: Dict[str, List[dict]] = {}   # {YYYYMMDD: [{symbol, symbol_name, volume}, ...]}
    adaptive_mode = (score_method or "").lower() == "adaptive_blend"
    if adaptive_methods is None:
        adaptive_methods = ["recent_ic_blend", "train_ic_blend", "ic_blend", "xgb"]
    adaptive_methods = [m.strip() for m in adaptive_methods if m and m.strip()]
    adaptive_history: Dict[str, List[float]] = {m: [] for m in adaptive_methods}
    risk_peak = capital
    risk_cooldown_remaining = 0

    logger.info("[Step 4] 开始逐日回测...")
    logger.info("-" * 70)

    for trade_date in trade_dates:
        prev_dates = [d for d in trade_dates if d < trade_date]
        if not prev_dates:
            continue
        as_of_date = prev_dates[-1]

        date_str = trade_date.strftime("%Y%m%d")

        if risk_cooldown_remaining > 0:
            selected_method = "risk_off"
            decisions = []
            next_capital = capital
            records = []
            detail_df = pd.DataFrame()
            risk_cooldown_remaining -= 1
        elif adaptive_mode:
            selected_method = _select_adaptive_method(
                adaptive_history,
                lookback=adaptive_lookback,
                min_observations=adaptive_min_observations,
                cash_threshold=adaptive_cash_threshold,
                default_method=adaptive_default_method,
                switch_margin=adaptive_switch_margin,
            )
            method_results = {}
            for method in adaptive_methods:
                method_decisions = _build_decisions_for_method(
                    screener=screener,
                    price_map=price_map,
                    as_of_date=as_of_date,
                    stock_info=tradable_info,
                    top_n=top_n,
                    factor_cache_dir=factor_cache_dir,
                    score_method=method,
                    available_capital=capital,
                    position_count=position_count,
                    confidence_threshold=confidence_threshold,
                    fill_positions=fill_positions,
                    allocation_method=allocation_method,
                    sizer=sizer,
                    max_stock_price=max_stock_price,
                )
                method_capital, method_records, method_detail = settle_day(
                    method_decisions,
                    price_map,
                    trade_date,
                    capital,
                )
                method_return = (method_capital - capital) / capital if capital > 0 else 0.0
                adaptive_history[method].append(method_return)
                method_results[method] = (method_capital, method_records, method_detail, method_decisions)

            if selected_method == "cash":
                decisions = []
                next_capital = capital
                records = []
                detail_df = pd.DataFrame()
            else:
                next_capital, records, detail_df, decisions = method_results.get(
                    selected_method,
                    (capital, [], pd.DataFrame(), []),
                )
        else:
            selected_method = score_method
            decisions = _build_decisions_for_method(
                screener=screener,
                price_map=price_map,
                as_of_date=as_of_date,
                stock_info=tradable_info,
                top_n=top_n,
                factor_cache_dir=factor_cache_dir,
                score_method=score_method,
                available_capital=capital,
                position_count=position_count,
                confidence_threshold=confidence_threshold,
                fill_positions=fill_positions,
                allocation_method=allocation_method,
                sizer=sizer,
                max_stock_price=max_stock_price,
            )
            next_capital, records, detail_df = settle_day(decisions, price_map, trade_date, capital)

        # 记录当日操作建议（赛制 JSON 格式：symbol / symbol_name / volume）
        advice = formatter.format(decisions)
        daily_advice[date_str] = advice

        # 终端实时输出当日操作建议 JSON（便于在 VSCode 终端直观查看 / 录屏）
        if print_advice:
            print(f"\n===== {date_str} 操作建议（共 {len(advice)} 只）=====", flush=True)
            print(json.dumps(advice, ensure_ascii=False, indent=2), flush=True)

        day_pnl = next_capital - capital
        ret_pct = day_pnl / capital * 100 if capital > 0 else 0.0

        logger.info("%s  方法 %s | 持仓 %d 只 | 日盈亏 %+.2f | 日收益 %+.2f%% | 总资产 %.2f",
                    trade_date.strftime("%Y-%m-%d"), selected_method, len(records), day_pnl, ret_pct, next_capital)
        if not detail_df.empty:
            for _, row in detail_df.iterrows():
                logger.info("    %-8s %-10s 昨收%.2f 今收%.2f 金额%.0f 盈亏%+.2f(%+.2f%%)",
                            row["symbol"], row["symbol_name"],
                            row["prev_close"], row["today_close"],
                            row["amount"], row["pnl"], row["pnl_pct"])

        daily_records.append({"date": trade_date, "total_assets": next_capital,
                               "pnl": day_pnl, "positions": len(records),
                               "return_pct": ret_pct, "method": selected_method})
        capital = next_capital
        risk_peak = max(risk_peak, capital)
        if max_drawdown_stop and max_drawdown_stop > 0 and risk_peak > 0:
            current_drawdown = (risk_peak - capital) / risk_peak
            if current_drawdown >= max_drawdown_stop and risk_cooldown_remaining <= 0:
                risk_cooldown_remaining = risk_cooldown_days if risk_cooldown_days > 0 else len(trade_dates)
                risk_peak = capital
                logger.warning(
                    "Risk stop triggered: drawdown %.2f%% >= %.2f%%; cash for %d trading days",
                    current_drawdown * 100,
                    max_drawdown_stop * 100,
                    risk_cooldown_remaining,
                )

    # ---- Step 5: 汇总 ----
    result_df = pd.DataFrame(daily_records).set_index("date")
    total_return = (capital - initial_capital) / initial_capital * 100
    max_dd = _max_drawdown(result_df["total_assets"].values)

    logger.info("=" * 60)
    logger.info("回测结果汇总")
    logger.info("  初始资金:   %.2f", initial_capital)
    logger.info("  期末资产:   %.2f", capital)
    logger.info("  累计收益率: %+.2f%%", total_return)
    logger.info("  最大回撤:   %.2f%%", max_dd * 100)
    logger.info("  交易天数:   %d", len(result_df))
    logger.info("  盈利天数:   %d / 亏损天数: %d",
                (result_df["pnl"] > 0).sum(), (result_df["pnl"] < 0).sum())
    logger.info("=" * 60)

    os.makedirs(log_dir, exist_ok=True)
    tag = f"{start_date}_{end_date}" + (f"_s{max_stocks}" if max_stocks else "")
    csv_path = os.path.join(log_dir, f"backtest_{tag}.csv")
    result_df.to_csv(csv_path, encoding="utf-8-sig")
    logger.info("明细已保存: %s", csv_path)

    # ---- 每日操作建议 JSON ----
    # 1) 每个交易日单独一份: logs/advice/YYYYMMDD.json
    advice_dir = os.path.join(log_dir, f"advice_{tag}")
    os.makedirs(advice_dir, exist_ok=True)
    for d_str, advice in daily_advice.items():
        with open(os.path.join(advice_dir, f"{d_str}.json"), "w", encoding="utf-8") as f:
            json.dump(advice, f, ensure_ascii=False, indent=2)
    # 2) 汇总一份: {日期: [建议...]}
    all_path = os.path.join(log_dir, f"advice_{tag}.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(daily_advice, f, ensure_ascii=False, indent=2)
    logger.info("每日操作建议已保存: %s/（共 %d 天）", advice_dir, len(daily_advice))
    logger.info("操作建议汇总: %s", all_path)

    if no_plot:
        logger.info("已跳过图表生成 (--no-plot)")
    else:
        _plot_results(result_df, initial_capital, log_dir, tag)
    return result_df


def _max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _plot_results(df: pd.DataFrame, initial_capital: float, log_dir: str, tag: str) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from matplotlib.ticker import FuncFormatter

        # 字体：中文宋体、英文 Times New Roman
        matplotlib.rcParams["axes.unicode_minus"] = False
        zh_font  = fm.FontProperties(family=["SimSun", "宋体", "STSong", "serif"])
        en_font  = fm.FontProperties(family=["Times New Roman", "serif"])

        # 配色
        C_BLUE   = "#185FA5"   # 总资产曲线
        C_BG     = "#FAFAF7"   # 米白背景
        C_GRID   = "#E8E6DF"   # 浅灰网格
        C_GRAY   = "#888780"   # 参考线 / 次要文字
        C_GREEN  = "#3B6D11"   # 亏损（A股习惯：绿跌）
        C_RED    = "#A32D2D"   # 盈利（A股习惯：红涨）
        C_AMBER  = "#BA7517"   # 累计收益率曲线

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 8),
            gridspec_kw={"height_ratios": [3, 2]},
            facecolor=C_BG,
        )
        fig.subplots_adjust(hspace=0.45)

        dates = df.index
        assets  = df["total_assets"] / 10000
        cum_ret = (df["total_assets"] / initial_capital - 1) * 100
        bar_clr = [C_RED if v >= 0 else C_GREEN for v in df["pnl"]]

        # ── 上图：总资产 ──────────────────────────────────────────────
        ax1.set_facecolor(C_BG)
        ax1.fill_between(dates, assets, initial_capital / 10000,
                         where=(assets >= initial_capital / 10000),
                         alpha=0.12, color=C_BLUE, interpolate=True)
        ax1.fill_between(dates, assets, initial_capital / 10000,
                         where=(assets < initial_capital / 10000),
                         alpha=0.12, color=C_GREEN, interpolate=True)
        ax1.plot(dates, assets, color=C_BLUE, lw=2, zorder=3)
        ax1.scatter(dates, assets, color=C_BLUE, s=40, zorder=4)
        ax1.axhline(initial_capital / 10000, color=C_GRAY,
                    lw=1, linestyle="--", alpha=0.7, label="初始资金基准")

        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax1.set_ylabel("总资产（万元）", fontproperties=zh_font, fontsize=11, color=C_GRAY)
        ax1.tick_params(colors=C_GRAY, labelsize=9)
        for lbl in ax1.get_xticklabels():
            lbl.set_fontproperties(en_font)
        for lbl in ax1.get_yticklabels():
            lbl.set_fontproperties(en_font)
        ax1.grid(axis="y", color=C_GRID, lw=0.8)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.spines[["left", "bottom"]].set_color(C_GRID)

        legend = ax1.legend(prop=zh_font, fontsize=9, framealpha=0,
                            labelcolor=C_GRAY, loc="upper left")

        title_parts = tag.replace("_", "  ").replace("s", "样本")
        ax1.set_title(f"竞赛回测  {title_parts}",
                      fontproperties=zh_font, fontsize=13,
                      color="#2C2C2A", pad=12)

        # ── 下图：日收益柱 + 累计收益线 ────────────────────────────────
        ax2.set_facecolor(C_BG)
        ax2.bar(dates, df["return_pct"], color=bar_clr, alpha=0.78,
                width=0.6, zorder=2, label="日收益率")
        ax2_r = ax2.twinx()
        ax2_r.plot(dates, cum_ret, color=C_AMBER, lw=1.8,
                   marker="D", ms=4, zorder=3, label="累计收益率")
        ax2_r.set_facecolor(C_BG)
        ax2_r.tick_params(colors=C_GRAY, labelsize=9)
        for lbl in ax2_r.get_yticklabels():
            lbl.set_fontproperties(en_font)
        ax2_r.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax2_r.spines[["top"]].set_visible(False)
        ax2_r.spines[["right", "left", "bottom"]].set_color(C_GRID)

        ax2.axhline(0, color=C_GRAY, lw=0.8, alpha=0.6)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax2.set_ylabel("日收益率", fontproperties=zh_font, fontsize=10, color=C_GRAY)
        ax2_r.set_ylabel("累计收益率", fontproperties=zh_font, fontsize=10, color=C_AMBER)
        ax2.tick_params(colors=C_GRAY, labelsize=9)
        for lbl in ax2.get_xticklabels():
            lbl.set_fontproperties(en_font)
        for lbl in ax2.get_yticklabels():
            lbl.set_fontproperties(en_font)
        ax2.grid(axis="y", color=C_GRID, lw=0.8)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.spines[["left", "bottom"]].set_color(C_GRID)

        # 合并图例
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2_r.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, prop=zh_font, fontsize=9,
                   framealpha=0, labelcolor=C_GRAY, loc="upper left")

        # 统计注释
        total_ret = (df["total_assets"].iloc[-1] - initial_capital) / initial_capital * 100
        win_days  = (df["pnl"] > 0).sum()
        note = (f"累计收益 {total_ret:+.2f}%   "
                f"胜率 {win_days}/{len(df)}   "
                f"最大回撤 {_max_drawdown(df['total_assets'].values)*100:.2f}%")
        fig.text(0.99, 0.01, note, ha="right", va="bottom",
                 fontproperties=zh_font, fontsize=9, color=C_GRAY)

        png_path = os.path.join(log_dir, f"backtest_{tag}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight",
                    facecolor=C_BG, edgecolor="none")
        plt.close(fig)
        logger.info("图表已保存: %s", png_path)
    except Exception as e:
        logger.warning("图表生成失败: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="竞赛口径量化回测（支持断点续传）")
    parser.add_argument("--mvp", action="store_true",
                        help="MVP 快速测试：500 只股票 × 最近 3 个交易日")
    parser.add_argument("--start", type=str, default="20260518", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", type=str,
                        default=(datetime.now().strftime("%Y%m%d")),
                        help="结束日期 YYYYMMDD（默认今天）")
    parser.add_argument("--capital", type=float, default=500000.0, help="初始资金")
    parser.add_argument("--top-n", type=int, default=20, help="每日候选池大小")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="限制股票池大小（用于快速测试）")
    parser.add_argument("--interval", type=float, default=0.2, help="请求间隔秒")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式：完全不联网，仅用本地缓存的股票回测（未缓存的跳过）")
    parser.add_argument("--cache-dir", type=str, default="data/prefetch_cache",
                        help="K 线缓存目录（断点续传用）")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志输出目录")
    parser.add_argument("--model-path", type=str, default="data/xgb_scorer.model",
                        help="XGBoost 模型路径（不存在时自动降级为等权打分）")
    parser.add_argument("--factor-cache-dir", type=str, default=None,
                        help="每日截面因子缓存目录；重复回测同一股票池/日期时可显著提速")
    parser.add_argument("--no-plot", action="store_true",
                        help="跳过回测图表生成，加快长周期批量回测")
    parser.add_argument(
        "--score-method",
        choices=["xgb", "ic_blend", "recent_ic_blend", "train_ic_blend", "ensemble_blend", "adaptive_blend"],
        default="train_ic_blend",
        help="Ranking method",
    )
    parser.add_argument("--position-count", type=int, default=3,
                        help="Target number of daily positions")
    parser.add_argument("--confidence-threshold", type=float, default=0.55,
                        help="Confidence threshold; set negative to disable")
    parser.add_argument("--no-fill-positions", action="store_true",
                        help="Do not fill below-threshold candidates up to position-count")
    parser.add_argument("--allocation-method", choices=["equal", "score"], default="score",
                        help="Capital allocation method")
    parser.add_argument("--quiet-advice", action="store_true",
                        help="Do not print daily advice JSON to stdout")
    parser.add_argument("--adaptive-methods", type=str,
                        default="recent_ic_blend,train_ic_blend,ic_blend,xgb",
                        help="Comma-separated methods used by adaptive_blend")
    parser.add_argument("--adaptive-lookback", type=int, default=5,
                        help="Trailing days used by adaptive_blend")
    parser.add_argument("--adaptive-min-observations", type=int, default=3,
                        help="Minimum method observations before adaptive_blend switches")
    parser.add_argument("--adaptive-cash-threshold", type=float, default=-0.002,
                        help="Use cash if best trailing average return is below this value")
    parser.add_argument("--adaptive-default-method", type=str, default="recent_ic_blend",
                        help="Method used before adaptive_blend has enough history")
    parser.add_argument("--adaptive-switch-margin", type=float, default=0.003,
                        help="Required trailing-return advantage before switching from default method")
    parser.add_argument("--max-drawdown-stop", type=float, default=0,
                        help="Portfolio drawdown threshold that triggers risk-off; set 0 to disable")
    parser.add_argument("--risk-cooldown-days", type=int, default=10,
                        help="Trading days to stay in cash after the drawdown stop is triggered")
    parser.add_argument("--max-stock-price", type=float, default=0,
                        help="Skip candidates above this previous close price; set 0 to disable")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    confidence_threshold = None if args.confidence_threshold < 0 else args.confidence_threshold
    max_stock_price = None if args.max_stock_price <= 0 else args.max_stock_price

    if args.mvp:
        # MVP：最近 3 个交易日，500 只股票
        end = datetime.now().strftime("%Y%m%d")
        # 往前推约 7 天保证拿到 3 个交易日
        start = (datetime.now() - pd.Timedelta(days=7)).strftime("%Y%m%d")
        run_backtest(
            start_date=start,
            end_date=end,
            initial_capital=args.capital,
            top_n=args.top_n,
            max_stocks=500,
            request_interval=args.interval,
            cache_dir=args.cache_dir,
            log_dir=args.log_dir,
            model_path=args.model_path,
            offline=args.offline,
            factor_cache_dir=args.factor_cache_dir,
            no_plot=args.no_plot,
            score_method=args.score_method,
            position_count=args.position_count,
            confidence_threshold=confidence_threshold,
            fill_positions=not args.no_fill_positions,
            allocation_method=args.allocation_method,
            print_advice=not args.quiet_advice,
            adaptive_methods=args.adaptive_methods.split(","),
            adaptive_lookback=args.adaptive_lookback,
            adaptive_min_observations=args.adaptive_min_observations,
            adaptive_cash_threshold=args.adaptive_cash_threshold,
            adaptive_default_method=args.adaptive_default_method,
            adaptive_switch_margin=args.adaptive_switch_margin,
            max_drawdown_stop=args.max_drawdown_stop,
            risk_cooldown_days=args.risk_cooldown_days,
            max_stock_price=max_stock_price,
        )
    else:
        run_backtest(
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            top_n=args.top_n,
            max_stocks=args.max_stocks,
            request_interval=args.interval,
            cache_dir=args.cache_dir,
            log_dir=args.log_dir,
            model_path=args.model_path,
            offline=args.offline,
            factor_cache_dir=args.factor_cache_dir,
            no_plot=args.no_plot,
            score_method=args.score_method,
            position_count=args.position_count,
            confidence_threshold=confidence_threshold,
            fill_positions=not args.no_fill_positions,
            allocation_method=args.allocation_method,
            print_advice=not args.quiet_advice,
            adaptive_methods=args.adaptive_methods.split(","),
            adaptive_lookback=args.adaptive_lookback,
            adaptive_min_observations=args.adaptive_min_observations,
            adaptive_cash_threshold=args.adaptive_cash_threshold,
            adaptive_default_method=args.adaptive_default_method,
            adaptive_switch_margin=args.adaptive_switch_margin,
            max_drawdown_stop=args.max_drawdown_stop,
            risk_cooldown_days=args.risk_cooldown_days,
            max_stock_price=max_stock_price,
        )
