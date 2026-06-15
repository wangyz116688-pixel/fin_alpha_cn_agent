"""
时序特征算子模块（AlphaNet-style）。

实现 6 类向量化时序算子，并基于原始输入序列生成 10 个组合特征。

算子:
    ts_corr(X, Y, d)       - Pearson 相关系数（滚动窗口）
    ts_cov(X, Y, d)        - 协方差（滚动窗口）
    ts_stddev(X, d)        - 标准差（滚动窗口）
    ts_zscore(X, d)        - Z-score（滚动窗口）
    ts_return(X, d)        - 涨跌幅（d 日）
    ts_decaylinear(X, d)   - 线性衰减加权均值

对外接口:
    compute_alphanet_features(df, symbol) -> pd.Series
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 6 类时序算子（向量化，接受 pd.Series / np.ndarray）
# ---------------------------------------------------------------------------


def ts_corr(
    X: pd.Series,
    Y: pd.Series,
    d: int,
) -> float:
    """X 与 Y 过去 d 天的 Pearson 相关系数（取最新窗口值）。"""
    n = len(X)
    if n < d or d < 3:
        return np.nan
    return X.iloc[-d:].corr(Y.iloc[-d:])


def ts_cov(
    X: pd.Series,
    Y: pd.Series,
    d: int,
) -> float:
    """X 与 Y 过去 d 天的协方差（取最新窗口值）。"""
    n = len(X)
    if n < d or d < 3:
        return np.nan
    return X.iloc[-d:].cov(Y.iloc[-d:])


def ts_stddev(
    X: pd.Series,
    d: int,
) -> float:
    """X 过去 d 天的标准差。"""
    n = len(X)
    if n < d or d < 2:
        return np.nan
    return X.iloc[-d:].std(ddof=1)  # 样本标准差


def ts_zscore(
    X: pd.Series,
    d: int,
) -> float:
    """X 过去 d 天的 Z-score（最新值相对于窗口均值和标准差的偏离）。"""
    n = len(X)
    if n < d or d < 2:
        return np.nan
    window = X.iloc[-d:]
    mean = window.mean()
    std = window.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return (X.iloc[-1] - mean) / std


def ts_return(
    X: pd.Series,
    d: int,
) -> float:
    """X 过去 d 天的涨跌幅：(X[-1] / X[-d-1] - 1)。"""
    n = len(X)
    if n < d + 1:
        return np.nan
    prev = X.iloc[-(d + 1)]
    curr = X.iloc[-1]
    if prev == 0 or np.isnan(prev):
        return np.nan
    return (curr / prev) - 1.0


def ts_decaylinear(
    X: pd.Series,
    d: int,
) -> float:
    """线性衰减加权均值：权重 w_i = (d - i) / sum_{j=0}^{d-1} (d - j)，近期权重大。"""
    n = len(X)
    if n < d or d < 2:
        return np.nan
    window = X.iloc[-d:].values.astype(float)
    if np.any(np.isnan(window)):
        return np.nan
    # 权重：最近一天权重 = d，最早一天权重 = 1
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    return float(np.dot(window, weights))


# ---------------------------------------------------------------------------
# 批量计算算子（返回向量，用于逐行特征生成）
# ---------------------------------------------------------------------------


def _rolling_corr(X: pd.Series, Y: pd.Series, d: int) -> pd.Series:
    """滚动 Pearson 相关系数序列。"""
    return X.rolling(d, min_periods=d).corr(Y)


def _rolling_cov(X: pd.Series, Y: pd.Series, d: int) -> pd.Series:
    """滚动协方差序列。"""
    return X.rolling(d, min_periods=d).cov(Y)


def _rolling_stddev(X: pd.Series, d: int) -> pd.Series:
    """滚动标准差序列。"""
    return X.rolling(d, min_periods=d).std(ddof=1)


def _rolling_zscore(X: pd.Series, d: int) -> pd.Series:
    """滚动 Z-score 序列。"""
    mean = X.rolling(d, min_periods=d).mean()
    std = X.rolling(d, min_periods=d).std(ddof=1)
    result = (X - mean) / std.replace(0, np.nan)
    return result.fillna(0.0)


def _rolling_return(X: pd.Series, d: int) -> pd.Series:
    """滚动 d 日涨跌幅序列。"""
    return X.pct_change(periods=d)


def _rolling_decaylinear(X: pd.Series, d: int) -> pd.Series:
    """滚动线性衰减加权均值序列。"""
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    return X.rolling(d, min_periods=d).apply(
        lambda w: float(np.dot(w, weights)), raw=True
    )


# ---------------------------------------------------------------------------
# 10 个组合特征（基于原始序列按 PROJECT_SPEC 3.3 生成）
# ---------------------------------------------------------------------------

# 固定配置：哪些特征用哪些算子/输入
_FEATURE_SPECS = [
    # (特征名, 算子函数, 参数)
    ("corr_close_vol_5", _rolling_corr, {"X_col": "close", "Y_col": "volume", "d": 5}),
    ("corr_close_turn_5", _rolling_corr, {"X_col": "close", "Y_col": "turn", "d": 5}),
    ("corr_return_vol_5", _rolling_corr, {"X_col": "return_1d", "Y_col": "volume", "d": 5}),
    ("std_close_10", _rolling_stddev, {"X_col": "close", "d": 10}),
    ("std_vol_10", _rolling_stddev, {"X_col": "volume", "d": 10}),
    ("zscore_close_10", _rolling_zscore, {"X_col": "close", "d": 10}),
    ("zscore_vol_5", _rolling_zscore, {"X_col": "volume", "d": 5}),
    ("ret_close_10", _rolling_return, {"X_col": "close", "d": 10}),
    ("decay_vol_5", _rolling_decaylinear, {"X_col": "volume", "d": 5}),
    ("decay_close_5", _rolling_decaylinear, {"X_col": "close", "d": 5}),
]


def compute_alphanet_features(
    df: pd.DataFrame,
    symbol: str,
) -> Optional[pd.Series]:
    """
    对单只股票的标准化 K 线 DataFrame，计算 10 个 AlphaNet 组合特征，
    返回最新一行（T-1）的 Series。

    Parameters
    ----------
    df : pd.DataFrame
        需包含列: open, high, low, close, volume, amount, turn
        可选列: date（用于排序）
    symbol : str
        股票代码。

    Returns
    -------
    pd.Series or None
        10 个特征值，若数据不足则返回 None。
    """
    # 构建原始输入序列
    raw = _prepare_input_series(df)
    if raw is None:
        logger.warning("%s 数据不足以计算 AlphaNet 特征", symbol)
        return None

    result = {"symbol": symbol}
    for feat_name, func, kwargs in _FEATURE_SPECS:
        # 展开参数
        params = kwargs.copy()
        if "X_col" in params:
            params["X"] = raw[params.pop("X_col")]
        if "Y_col" in params:
            params["Y"] = raw[params.pop("Y_col")]
        d = params.pop("d")

        try:
            all_vals = func(**params, d=d)
            result[feat_name] = all_vals.iloc[-1] if len(all_vals) > 0 else np.nan
        except Exception:
            logger.warning("%s 计算特征 %s 失败", symbol, feat_name)
            result[feat_name] = np.nan

    return pd.Series(result)


def _prepare_input_series(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    准备原始输入序列 DataFrame，含 9 个序列：
    open, high, low, close, volume, amount, turn, vwap, return_1d
    """
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    required = ["open", "high", "low", "close", "volume", "amount", "turn"]
    for col in required:
        if col not in df.columns:
            logger.warning("缺少列 %s", col)
            return None

    n = len(df)
    if n < 12:  # 至少要支持 10 日窗口
        return None

    out = pd.DataFrame(index=df.index)
    out["open"] = df["open"].astype(float)
    out["high"] = df["high"].astype(float)
    out["low"] = df["low"].astype(float)
    out["close"] = df["close"].astype(float)
    out["volume"] = df["volume"].astype(float)
    out["amount"] = df["amount"].astype(float)
    out["turn"] = df["turn"].astype(float)

    # vwap = amount / (volume * 100)，成交量单位手 → 股
    vol_shares = out["volume"] * 100
    out["vwap"] = np.where(vol_shares != 0, out["amount"] / vol_shares, np.nan)

    # return_1d = close / close.shift(1) - 1
    out["return_1d"] = out["close"].pct_change()

    return out