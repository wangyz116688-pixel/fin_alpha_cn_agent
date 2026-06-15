"""
五因子 OLS 中性化模块。

对因子截面数据进行行业 + 风格中性化，消除行业偏差和风格暴露，
提升因子 IC 稳定性。

方法：每日截面数据，以行业/市值/动量/波动率/换手率为自变量做 OLS 回归，
取残差作为中性化后的因子值。

对外接口:
    neutralize_factors(factor_df, industry_map, market_cap_map) -> pd.DataFrame
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# 申万一级 28 行业（标准分类）
SW_INDUSTRY_28 = [
    "农林牧渔", "采掘", "化工", "钢铁", "有色金属",
    "电子", "家用电器", "食品饮料", "纺织服装", "轻工制造",
    "医药生物", "公用事业", "交通运输", "房地产", "商业贸易",
    "休闲服务", "综合", "建筑材料", "建筑装饰", "电气设备",
    "国防军工", "计算机", "传媒", "通信", "银行",
    "非银金融", "汽车", "机械设备",
]


def neutralize_factors(
    factor_df: pd.DataFrame,
    industry_map: Optional[Dict[str, str]] = None,
    market_cap_map: Optional[Dict[str, float]] = None,
    mom_20d_col: str = "mom_20d",
    vol_20d_col: str = "vol_20d",
    turn_20d_col: Optional[str] = None,
    turn_col: str = "turn_5d_avg",
    factor_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    对因子 DataFrame 进行五因子 OLS 截面中性化。

    Parameters
    ----------
    factor_df : pd.DataFrame
        因子数据，行索引为 symbol，列为各因子值。
        需包含 mom_20d、vol_20d 列（如果 turn_20d_col 未指定则回退到 turn_col）。
    industry_map : dict, optional
        {symbol: 行业名} 映射。若未提供则跳过行业哑变量。
    market_cap_map : dict, optional
        {symbol: 市值（亿）} 映射。若未提供则跳过市值变量。
    mom_20d_col : str
        20 日动量列名，默认 "mom_20d"。
    vol_20d_col : str
        20 日波动率列名，默认 "vol_20d"。
    turn_20d_col : str, optional
        20 日平均换手率列名。若为 None 则回退到 turn_col。
    turn_col : str
        换手率列名（当 turn_20d_col 未提供时使用），默认 "turn_5d_avg"。
    factor_cols : list[str], optional
        需要中性化的因子列名列表。若为 None，则对 factor_df 中所有数值列做中性化。

    Returns
    -------
    pd.DataFrame
        中性化后的因子 DataFrame，行/列结构与输入一致（仅因子值替换为残差）。
    """
    if factor_df.empty:
        return factor_df

    df = factor_df.copy()
    # 将 inf/-inf 替换为 NaN，避免 OLS 拟合因无穷值报错（随后由 dropna 清理）
    df = df.replace([np.inf, -np.inf], np.nan)

    # 确定换手率列
    if turn_20d_col is None:
        turn_20d_col = turn_col
    effective_turn_col = turn_20d_col if turn_20d_col in df.columns else turn_col

    # 构建回归自变量 DataFrame，行索引 = symbol
    reg_df = pd.DataFrame(index=df.index)

    # 1. 行业哑变量
    if industry_map:
        industry_series = pd.Series(industry_map).reindex(df.index)
        # 将不在 28 行业中的归为 "其他"
        industry_series = industry_series.apply(
            lambda x: x if x in SW_INDUSTRY_28 else "其他"
        )
        industry_dummies = pd.get_dummies(industry_series, prefix="ind")
        reg_df = pd.concat([reg_df, industry_dummies], axis=1)

    # 2. 对数总市值
    if market_cap_map:
        cap_series = pd.Series(market_cap_map).reindex(df.index)
        # ln(market_cap)，市值单位亿，+1 防止 log(0)
        reg_df["ln_market_cap"] = np.log(cap_series.clip(lower=0.01))

    # 3. 20 日动量
    if mom_20d_col in df.columns:
        reg_df["mom_20d"] = df[mom_20d_col]
    else:
        logger.warning("缺少列 %s，跳过动量中性化", mom_20d_col)

    # 4. 20 日波动率
    if vol_20d_col in df.columns:
        reg_df["vol_20d"] = df[vol_20d_col]
    else:
        logger.warning("缺少列 %s，跳过波动率中性化", vol_20d_col)

    # 5. 20 日平均换手率
    if effective_turn_col in df.columns:
        reg_df["turn_avg"] = df[effective_turn_col]
    else:
        logger.warning("缺少列 %s，跳过换手率中性化", effective_turn_col)

    # 确定需要中性化的因子列
    if factor_cols is None:
        # 自动排除符号/名称列和非数值列
        exclude = {"symbol", "symbol_name", "industry", "market_cap_bn"}
        factor_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        # 也排除回归自变量本身
        reg_vars = set(reg_df.columns)
        factor_cols = [c for c in factor_cols if c not in reg_vars]

    if not factor_cols:
        logger.warning("没有可中性化的因子列")
        return df

    # 逐因子做 OLS 回归取残差
    for col in factor_cols:
        y = df[col].copy()
        # 构建训练数据：去除 y 和 reg 中有 NaN 的行
        combined = pd.concat([y, reg_df], axis=1).dropna()
        if len(combined) < max(3, reg_df.shape[1] + 1):
            # 样本太少，跳过该因子
            logger.warning("因子 %s 有效样本不足（%d），跳过中性化", col, len(combined))
            continue

        X = combined[reg_df.columns].values
        y_clean = combined[col].values

        try:
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y_clean)
            y_pred = model.predict(X)

            # 残差 = y - y_pred
            residuals = y_clean - y_pred

            # 将残差写回原始 df（保留 NaN 行）
            result_series = pd.Series(np.nan, index=df.index)
            result_series.loc[combined.index] = residuals
            df[col] = result_series
        except Exception:
            logger.exception("因子 %s OLS 中性化失败", col)

    return df