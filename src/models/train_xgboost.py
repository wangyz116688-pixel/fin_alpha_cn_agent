"""
XGBoost 选股模型训练脚本。

用历史全市场（或采样）K 线构建训练集：
  - 特征 X：截至 T 日的量价因子(13) + AlphaNet 特征(10)，经五因子中性化
  - 标签 y：T → T+1 的次日截面收益率，分 10 档（rank:pairwise 排序目标）
  - 严格防未来函数：T 日特征只用截至 T 日收盘的数据，标签是 T 之后才知道的收益

时序切分训练/验证（验证集时间严格晚于训练集），训练后保存模型，
供 StockScreener.set_model() 加载。

使用方式：
    # 快速冒烟（300 只 × 短区间）
    python -m src.models.train_xgboost --start 20240101 --end 20241231 --max-stocks 300

    # 完整训练
    python -m src.models.train_xgboost --start 20220101 --end 20241231 --output data/xgb_scorer.model
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 单日截面因子计算复用回测端的 compute_cross_section（量价 + AlphaNet + 中性化），
# 保证训练与回测/选股使用完全一致的特征。
# ---------------------------------------------------------------------------

def _close_on(price_map: Dict[str, pd.DataFrame], sym: str, date: pd.Timestamp) -> Optional[float]:
    df = price_map.get(sym)
    if df is None:
        return None
    rows = df[df["date"] == date]
    if rows.empty:
        return None
    return float(rows.iloc[-1]["close"])


# ---------------------------------------------------------------------------
# 构建训练集
# ---------------------------------------------------------------------------

def build_training_set(
    price_map: Dict[str, pd.DataFrame],
    screener,
    trade_dates: List[pd.Timestamp],
    feature_cols: List[str],
    lookback_days: int = 90,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    遍历每个交易日，拼接 (截面因子, 次日收益标签) 训练样本。

    Returns
    -------
    X : DataFrame, MultiIndex (date, symbol), 列为 feature_cols
    labels : Series, 同 index，值为 0~9 的分档标签
    """
    from src.models.xgboost_scorer import XGBoostScorer
    from src.backtest_stock_agent import compute_cross_section

    feat_frames = []
    ret_values = []
    idx_tuples = []

    # 最后一天没有"次日"，不能作为样本
    for i in range(len(trade_dates) - 1):
        t = trade_dates[i]
        t_next = trade_dates[i + 1]

        factor_df = compute_cross_section(price_map, t, screener, lookback_days)
        if factor_df.empty:
            continue

        # 仅保留模型特征列，缺失列补 NaN
        for col in feature_cols:
            if col not in factor_df.columns:
                factor_df[col] = np.nan
        factor_df = factor_df[feature_cols]

        # 次日收益标签
        day_feats = []
        for sym, row in factor_df.iterrows():
            c_t = _close_on(price_map, sym, t)
            c_next = _close_on(price_map, sym, t_next)
            if c_t is None or c_next is None or c_t <= 0:
                continue
            ret = c_next / c_t - 1.0
            day_feats.append(row.values)
            ret_values.append(ret)
            idx_tuples.append((t, sym))

        if day_feats:
            feat_frames.append(pd.DataFrame(day_feats, columns=feature_cols))

        if (i + 1) % 20 == 0:
            logger.info("  已处理 %d / %d 个交易日，累计样本 %d",
                        i + 1, len(trade_dates) - 1, len(idx_tuples))

    if not idx_tuples:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = pd.concat(feat_frames, ignore_index=True)
    X.index = pd.MultiIndex.from_tuples(idx_tuples, names=["date", "symbol"])

    returns = pd.Series(ret_values, index=X.index)
    labels = XGBoostScorer.build_labels(returns, n_buckets=10)

    # 丢弃标签为 NaN 的样本（当日有效股票数 < 10 档）
    valid = labels.notna()
    return X[valid], labels[valid]


# ---------------------------------------------------------------------------
# 时序切分 + group 构建
# ---------------------------------------------------------------------------

def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, pd.Series, np.ndarray]:
    """
    按日期时序切分：前 (1-val_ratio) 时间为训练，后段为验证。
    同时构建 rank:pairwise 所需的 group（每个交易日为一组）。
    """
    dates = sorted(X.index.get_level_values("date").unique())
    split_idx = int(len(dates) * (1 - val_ratio))
    train_dates = set(dates[:split_idx])
    val_dates = set(dates[split_idx:])

    train_mask = X.index.get_level_values("date").isin(train_dates)
    val_mask = X.index.get_level_values("date").isin(val_dates)

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]

    def _groups(frame: pd.DataFrame) -> np.ndarray:
        # 按日期排序后，每个交易日的样本数
        sizes = frame.groupby(level="date").size()
        return sizes.values

    # group 要求数据按日期连续排列
    X_tr = X_tr.sort_index(level="date")
    y_tr = y_tr.reindex(X_tr.index)
    X_va = X_va.sort_index(level="date")
    y_va = y_va.reindex(X_va.index)

    return X_tr, y_tr, _groups(X_tr), X_va, y_va, _groups(X_va)


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train(
    start_date: str,
    end_date: str,
    max_stocks: Optional[int] = None,
    val_ratio: float = 0.2,
    lookback_days: int = 90,
    output: str = "data/xgb_scorer.model",
    cache_dir: str = "data/train_cache",
    request_interval: float = 0.2,
) -> None:
    from src.backtest_stock_agent import prefetch_price_data
    from src.models.stock_screener import StockScreener
    from src.models.xgboost_scorer import XGBoostScorer

    logger.info("=" * 60)
    logger.info("XGBoost 训练  %s ~ %s  股票上限 %s",
                start_date, end_date, str(max_stocks) if max_stocks else "全市场")
    logger.info("=" * 60)

    # ---- 股票列表 ----
    screener = StockScreener(end_date=end_date, request_interval=request_interval)
    feature_cols = screener.all_factor_cols

    stock_info = screener.get_stock_list()
    if stock_info.empty:
        logger.error("获取股票列表失败")
        return
    tradable = screener.filter_tradable(stock_info)
    symbols = tradable.index.tolist()

    if max_stocks and max_stocks < len(symbols):
        import random
        random.seed(42)
        symbols = random.sample(symbols, max_stocks)
        logger.info("采样 %d 只股票用于训练", max_stocks)

    # ---- 预拉取 K 线（断点续传）----
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days + 40)).strftime("%Y%m%d")
    price_map = prefetch_price_data(
        symbols, fetch_start, end_date,
        request_interval=request_interval, cache_dir=cache_dir,
    )
    if not price_map:
        logger.error("K 线拉取失败")
        return

    # ---- 交易日序列（训练区间内）----
    all_dates: set = set()
    for df in price_map.values():
        all_dates.update(df["date"].tolist())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    trade_dates = sorted(d for d in all_dates if start_ts <= d <= end_ts)
    logger.info("训练区间交易日: %d 个", len(trade_dates))

    # ---- 构建训练集（带缓存：因子计算很慢，缓存后重训可秒级加载）----
    import os as _os
    import pickle
    ds_path = _os.path.join(
        cache_dir, f"trainset_{start_date}_{end_date}_s{max_stocks}_lb{lookback_days}.pkl"
    )
    if _os.path.exists(ds_path):
        logger.info("加载已缓存训练集: %s", ds_path)
        with open(ds_path, "rb") as f:
            X, y = pickle.load(f)
    else:
        logger.info("构建训练集（量价 + AlphaNet + 中性化 + 次日收益标签）...")
        X, y = build_training_set(price_map, screener, trade_dates, feature_cols, lookback_days)
        if not X.empty:
            _os.makedirs(cache_dir, exist_ok=True)
            with open(ds_path, "wb") as f:
                pickle.dump((X, y), f)
            logger.info("训练集已缓存: %s", ds_path)
    if X.empty:
        logger.error("训练集为空")
        return
    logger.info("训练样本: %d 行 × %d 特征", X.shape[0], X.shape[1])

    # 清理 inf/-inf（量价相关系数等因子在除零边界可能产生 inf），再填 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ---- 时序切分 ----
    X_tr, y_tr, g_tr, X_va, y_va, g_va = time_split(X, y, val_ratio)
    logger.info("训练集 %d 行 / %d 组 | 验证集 %d 行 / %d 组",
                len(X_tr), len(g_tr), len(X_va), len(g_va))

    # ---- 训练 ----
    scorer = XGBoostScorer(model_path=output)
    eval_set = [(X_va.values, y_va.values)] if len(X_va) > 0 else None
    eval_groups = [g_va] if len(X_va) > 0 else None

    scorer.train(
        X_tr.values,                # 统一用 numpy，避免与 eval_set 的列名不一致
        y_tr.values,
        groups=g_tr,
        eval_set=eval_set,
        eval_groups=eval_groups,
        early_stopping_rounds=20,
        verbose=True,
    )
    scorer.feature_names = feature_cols   # feature_names 单独保存

    # ---- 特征重要性 ----
    try:
        importance = scorer.model.get_score(importance_type="gain")
        # XGBoost 用 f0,f1... 命名，映射回因子名
        named = {}
        for k, v in importance.items():
            idx = int(k[1:]) if k.startswith("f") else None
            name = feature_cols[idx] if idx is not None and idx < len(feature_cols) else k
            named[name] = round(v, 2)
        top = sorted(named.items(), key=lambda x: x[1], reverse=True)
        logger.info("特征重要性（gain，Top 10）:")
        for name, val in top[:10]:
            logger.info("  %-22s %.2f", name, val)
    except Exception as e:
        logger.warning("特征重要性计算失败: %s", e)

    # ---- 保存 ----
    scorer.save(output)
    logger.info("=" * 60)
    logger.info("训练完成，模型已保存: %s", output)
    logger.info("在选股时加载: screener.set_model(XGBoostScorer()) 后 .load('%s')", output)
    logger.info("=" * 60)


def _parse_args():
    p = argparse.ArgumentParser(description="XGBoost 选股模型训练")
    p.add_argument("--start", type=str, required=True, help="训练数据起始 YYYYMMDD")
    p.add_argument("--end", type=str, required=True, help="训练数据结束 YYYYMMDD")
    p.add_argument("--max-stocks", type=int, default=None, help="采样股票数（加速）")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集时间占比")
    p.add_argument("--lookback-days", type=int, default=90, help="因子回看窗口天数")
    p.add_argument("--output", type=str, default="data/xgb_scorer.model", help="模型保存路径")
    p.add_argument("--cache-dir", type=str, default="data/train_cache", help="K 线缓存目录")
    p.add_argument("--interval", type=float, default=0.2, help="请求间隔秒")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        start_date=args.start,
        end_date=args.end,
        max_stocks=args.max_stocks,
        val_ratio=args.val_ratio,
        lookback_days=args.lookback_days,
        output=args.output,
        cache_dir=args.cache_dir,
        request_interval=args.interval,
    )
