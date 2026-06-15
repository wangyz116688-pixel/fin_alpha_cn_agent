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


# ---------------------------------------------------------------------------
# 断点续传：单股缓存
# ---------------------------------------------------------------------------

def _stock_cache_path(cache_dir: str, sym: str, start_date: str, end_date: str) -> str:
    return os.path.join(cache_dir, f"{sym}_{start_date}_{end_date}.pkl")


def _load_cached_stock(cache_dir: str, sym: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    path = _stock_cache_path(cache_dir, sym, start_date, end_date)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
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

    pv_rows = []
    alpha_rows = []
    for sym, df in price_map.items():
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
) -> List[dict]:
    factor_df = compute_cross_section(price_map, as_of_date, screener)
    if factor_df.empty:
        return []

    valid_syms = set(stock_info.index) & set(factor_df.index)
    factor_df = factor_df.loc[list(valid_syms)]
    if factor_df.empty:
        return []

    factor_df = screener._score(factor_df)
    candidates = screener._build_candidates(factor_df, stock_info, top_n)

    for c in candidates:
        if "C_mixed" not in c:
            c["C_mixed"] = c.get("xgb_score", 0.5)

    return candidates


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
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("竞赛回测  %s ~ %s  资金 %.0f  股票上限 %s",
                start_date, end_date, initial_capital,
                str(max_stocks) if max_stocks else "全市场")
    logger.info("=" * 60)

    # ---- Step 1: 股票列表 ----
    screener = StockScreener(end_date=end_date, request_interval=request_interval)
    screener.load_model(model_path)   # 不存在时自动降级为等权打分
    stock_info = screener.get_stock_list()
    if stock_info.empty:
        logger.error("获取股票列表失败")
        return pd.DataFrame()

    tradable_info = screener.filter_tradable(stock_info)
    symbols = tradable_info.index.tolist()

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
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=120)).strftime("%Y%m%d")
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

    sizer = PositionSizer()
    formatter = OutputFormatter(log_dir=log_dir)
    capital = initial_capital
    daily_records = []
    daily_advice: Dict[str, List[dict]] = {}   # {YYYYMMDD: [{symbol, symbol_name, volume}, ...]}

    logger.info("[Step 4] 开始逐日回测...")
    logger.info("-" * 70)

    for trade_date in trade_dates:
        prev_dates = [d for d in trade_dates if d < trade_date]
        if not prev_dates:
            continue
        as_of_date = prev_dates[-1]

        date_str = trade_date.strftime("%Y%m%d")

        candidates = screen_for_date(screener, price_map, as_of_date, tradable_info, top_n)
        if not candidates:
            logger.info("%s  空仓（无候选）", trade_date.date())
            daily_records.append({"date": trade_date, "total_assets": capital,
                                   "pnl": 0.0, "positions": 0, "return_pct": 0.0})
            daily_advice[date_str] = []
            continue

        candidates = sorted(candidates, key=lambda x: x.get("C_mixed", 0), reverse=True)
        high_conf = [c for c in candidates if c.get("C_mixed", 0) > 0.55]
        top5 = high_conf[:5] if high_conf else candidates[:5]

        prev_closes: Dict[str, float] = {}
        for c in top5:
            sym = c["symbol"]
            if sym in price_map:
                rows = price_map[sym][price_map[sym]["date"] == as_of_date]
                if not rows.empty:
                    prev_closes[sym] = float(rows.iloc[-1]["close"])

        decisions = sizer.allocate(top5, capital, prev_closes)

        # 记录当日操作建议（赛制 JSON 格式：symbol / symbol_name / volume）
        advice = formatter.format(decisions)
        daily_advice[date_str] = advice

        # 终端实时输出当日操作建议 JSON（便于在 VSCode 终端直观查看 / 录屏）
        print(f"\n===== {date_str} 操作建议（共 {len(advice)} 只）=====", flush=True)
        print(json.dumps(advice, ensure_ascii=False, indent=2), flush=True)

        next_capital, records, detail_df = settle_day(decisions, price_map, trade_date, capital)

        day_pnl = next_capital - capital
        ret_pct = day_pnl / capital * 100 if capital > 0 else 0.0

        logger.info("%s  持仓 %d 只 | 日盈亏 %+.2f | 日收益 %+.2f%% | 总资产 %.2f",
                    trade_date.strftime("%Y-%m-%d"), len(records), day_pnl, ret_pct, next_capital)
        if not detail_df.empty:
            for _, row in detail_df.iterrows():
                logger.info("    %-8s %-10s 昨收%.2f 今收%.2f 金额%.0f 盈亏%+.2f(%+.2f%%)",
                            row["symbol"], row["symbol_name"],
                            row["prev_close"], row["today_close"],
                            row["amount"], row["pnl"], row["pnl_pct"])

        daily_records.append({"date": trade_date, "total_assets": next_capital,
                               "pnl": day_pnl, "positions": len(records), "return_pct": ret_pct})
        capital = next_capital

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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

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
        )
