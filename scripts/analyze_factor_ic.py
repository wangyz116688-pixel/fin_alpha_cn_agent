from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.backtest_stock_agent import (
    _cached_symbols_covering,
    _stock_info_from_price_cache,
    compute_cross_section,
    prefetch_price_data,
)
from src.models.stock_screener import StockScreener


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _next_return(
    price_map: Dict[str, pd.DataFrame],
    symbol: str,
    as_of_date: pd.Timestamp,
    trade_date: pd.Timestamp,
) -> Optional[float]:
    df = price_map.get(symbol)
    if df is None or df.empty:
        return None

    prev = df[df["date"] == as_of_date]
    curr = df[df["date"] == trade_date]
    if prev.empty or curr.empty:
        return None

    prev_close = float(prev.iloc[-1]["close"])
    curr_close = float(curr.iloc[-1]["close"])
    if prev_close <= 0 or not np.isfinite(prev_close) or not np.isfinite(curr_close):
        return None
    return curr_close / prev_close - 1.0


def _spearman_ic(values: pd.Series, returns: pd.Series) -> float:
    data = pd.concat([values, returns], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        return np.nan
    if data.iloc[:, 0].nunique(dropna=True) < 3 or data.iloc[:, 1].nunique(dropna=True) < 3:
        return np.nan
    return float(data.iloc[:, 0].corr(data.iloc[:, 1], method="spearman"))


def _quantile_spread(values: pd.Series, returns: pd.Series, quantile: float) -> Tuple[float, float, float]:
    data = pd.concat([values, returns], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        return np.nan, np.nan, np.nan

    factor_col, ret_col = data.columns
    data = data.sort_values(factor_col, ascending=False)
    bucket = max(1, int(len(data) * quantile))
    top_ret = float(data.head(bucket)[ret_col].mean())
    bottom_ret = float(data.tail(bucket)[ret_col].mean())
    return top_ret, bottom_ret, top_ret - bottom_ret


def _load_universe(
    screener: StockScreener,
    cache_dir: str,
    fetch_start: str,
    end_date: str,
    offline: bool,
    max_stocks: Optional[int],
) -> pd.DataFrame:
    stock_info = screener.get_stock_list(offline=offline)
    if stock_info.empty and offline:
        stock_info = _stock_info_from_price_cache(cache_dir, fetch_start, end_date)
    if stock_info.empty:
        raise RuntimeError("stock universe is empty")

    tradable_info = screener.filter_tradable(stock_info)
    symbols = sorted(tradable_info.index.tolist())
    if offline:
        cached_symbols = _cached_symbols_covering(cache_dir, fetch_start, end_date)
        if cached_symbols:
            symbols = sorted(set(symbols) & cached_symbols)
            tradable_info = tradable_info.loc[tradable_info.index.isin(symbols)]
            logger.info("Offline cache-covered IC universe: %d symbols", len(symbols))
        else:
            logger.warning("Offline IC mode found no cache-covered symbols")
    if max_stocks and max_stocks < len(symbols):
        random.seed(42)
        symbols = sorted(random.sample(symbols, max_stocks))
        tradable_info = tradable_info.loc[symbols]
    else:
        tradable_info = tradable_info.loc[symbols]
    return tradable_info


def analyze_factor_ic(
    start_date: str,
    end_date: str,
    max_stocks: Optional[int] = None,
    cache_dir: str = "data/prefetch_cache",
    factor_cache_dir: str = "data/factor_cache",
    output_dir: str = "logs/factor_analysis",
    model_path: str = "data/xgb_scorer.model",
    offline: bool = True,
    quantile: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=120)).strftime("%Y%m%d")

    screener = StockScreener(end_date=end_date, request_interval=0)
    screener.load_model(model_path)

    tradable_info = _load_universe(
        screener=screener,
        cache_dir=cache_dir,
        fetch_start=fetch_start,
        end_date=end_date,
        offline=offline,
        max_stocks=max_stocks,
    )
    symbols = tradable_info.index.tolist()
    logger.info("IC universe: %d symbols", len(symbols))

    price_map = prefetch_price_data(
        symbols=symbols,
        start_date=fetch_start,
        end_date=end_date,
        request_interval=0,
        cache_dir=cache_dir,
        offline=offline,
    )
    if not price_map:
        raise RuntimeError("price data is empty")

    all_dates = sorted({d for df in price_map.values() for d in df["date"].tolist()})
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    trade_dates = [d for d in all_dates if start_ts <= d <= end_ts]
    if len(trade_dates) < 2:
        raise RuntimeError("not enough trade dates for IC analysis")

    daily_rows = []
    factor_cols = list(screener.all_factor_cols) + ["xgb_score"]

    for i in range(1, len(trade_dates)):
        as_of_date = trade_dates[i - 1]
        trade_date = trade_dates[i]
        factor_df = compute_cross_section(
            price_map=price_map,
            as_of_date=as_of_date,
            screener=screener,
            factor_cache_dir=factor_cache_dir,
        )
        valid_syms = sorted(set(tradable_info.index) & set(factor_df.index))
        factor_df = factor_df.loc[valid_syms] if valid_syms else pd.DataFrame()
        if factor_df.empty:
            continue

        factor_df = screener._score(factor_df)
        next_returns = {
            sym: _next_return(price_map, sym, as_of_date, trade_date)
            for sym in factor_df.index
        }
        ret = pd.Series(next_returns, name="next_return").dropna()
        factor_df = factor_df.loc[factor_df.index.intersection(ret.index)]
        ret = ret.loc[factor_df.index]
        if len(ret) < 20:
            continue

        for factor in factor_cols:
            if factor not in factor_df.columns:
                continue
            ic = _spearman_ic(factor_df[factor], ret)
            top_ret, bottom_ret, spread = _quantile_spread(factor_df[factor], ret, quantile)
            daily_rows.append(
                {
                    "as_of_date": as_of_date.date().isoformat(),
                    "trade_date": trade_date.date().isoformat(),
                    "factor": factor,
                    "universe": int(len(ret)),
                    "ic": ic,
                    "top_ret": top_ret,
                    "bottom_ret": bottom_ret,
                    "spread": spread,
                }
            )

    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        raise RuntimeError("no daily IC rows were produced")

    summary_rows = []
    for factor, group in daily_df.groupby("factor"):
        ic = group["ic"].dropna()
        spread = group["spread"].dropna()
        top_ret = group["top_ret"].dropna()
        bottom_ret = group["bottom_ret"].dropna()
        summary_rows.append(
            {
                "factor": factor,
                "days": int(group["trade_date"].nunique()),
                "avg_universe": float(group["universe"].mean()),
                "mean_ic": float(ic.mean()) if len(ic) else np.nan,
                "std_ic": float(ic.std(ddof=1)) if len(ic) > 1 else np.nan,
                "icir": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))) if len(ic) > 1 and ic.std(ddof=1) > 0 else np.nan,
                "positive_ic_rate": float((ic > 0).mean()) if len(ic) else np.nan,
                "mean_top_ret": float(top_ret.mean()) if len(top_ret) else np.nan,
                "mean_bottom_ret": float(bottom_ret.mean()) if len(bottom_ret) else np.nan,
                "mean_spread": float(spread.mean()) if len(spread) else np.nan,
                "spread_win_rate": float((spread > 0).mean()) if len(spread) else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_ic", "mean_spread"],
        ascending=[False, False],
    )

    os.makedirs(output_dir, exist_ok=True)
    tag = f"{start_date}_{end_date}" + (f"_s{max_stocks}" if max_stocks else "_all")
    daily_path = os.path.join(output_dir, f"factor_ic_daily_{tag}.csv")
    summary_path = os.path.join(output_dir, f"factor_ic_summary_{tag}.csv")
    json_path = os.path.join(output_dir, f"factor_ic_summary_{tag}.json")

    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.replace({np.nan: None}).to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    logger.info("Daily IC saved: %s", daily_path)
    logger.info("Summary IC saved: %s", summary_path)
    logger.info("Summary JSON saved: %s", json_path)
    return daily_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze factor IC and top-bottom return spread.")
    parser.add_argument("--start", required=True, help="Backtest start date, e.g. 20260515")
    parser.add_argument("--end", required=True, help="Backtest end date, e.g. 20260615")
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--cache-dir", default="data/prefetch_cache")
    parser.add_argument("--factor-cache-dir", default="data/factor_cache")
    parser.add_argument("--output-dir", default="logs/factor_analysis")
    parser.add_argument("--model-path", default="data/xgb_scorer.model")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--offline", action="store_true", help="Use local cached data only.")
    args = parser.parse_args()

    _, summary_df = analyze_factor_ic(
        start_date=args.start,
        end_date=args.end,
        max_stocks=args.max_stocks,
        cache_dir=args.cache_dir,
        factor_cache_dir=args.factor_cache_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        offline=args.offline,
        quantile=args.quantile,
    )

    cols = ["factor", "days", "mean_ic", "positive_ic_rate", "mean_spread", "spread_win_rate"]
    print(summary_df[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
