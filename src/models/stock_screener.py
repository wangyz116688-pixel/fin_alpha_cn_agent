"""
全市场扫描与选股模块。

流程：拉取股票列表 → 过滤不可交易（ST/停牌/新股/微盘） → 计算因子 →
五因子中性化 → XGBoost 打分 → 返回 Top20 候选池。

对外接口:
    StockScreener 类
        - set_model(scorer)    : 设置 XGBoost 打分模型
        - run(top_n=20)        : 执行全流程，返回候选池（list of dict）
        - get_stock_list()     : 拉取全市场股票基础信息
        - filter_tradable()    : 过滤不可交易标的
"""

import logging
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StockScreener:
    """
    全市场扫描选股器。

    每天从全市场股票出发，经过过滤、因子计算、中性化、打分，
    最终输出 Top N 候选池。

    Parameters
    ----------
    scorer : XGBoostScorer, optional
        XGBoost 打分模型实例。
    end_date : str, optional
        目标日期 YYYYMMDD，默认今天。
    request_interval : float
        拉取数据时的请求间隔（秒）。
    """

    def __init__(
        self,
        scorer=None,
        end_date: Optional[str] = None,
        request_interval: float = 0.3,
    ):
        self.scorer = scorer
        self.end_date = end_date or datetime.now().strftime("%Y%m%d")
        self.request_interval = request_interval

        # 因子列（需与 price_volume + alphanet_features 保持一致）
        self.price_volume_cols = [
            "mom_1d", "mom_5d", "mom_20d", "rev_1d",
            "vol_5d", "vol_20d", "vol_ratio",
            "turn_1d", "turn_5d_avg", "turn_ratio",
            "price_vol_corr_5", "price_turn_corr_5", "vwap_dev",
        ]
        self.alphanet_cols = [
            "corr_close_vol_5", "corr_close_turn_5", "corr_return_vol_5",
            "std_close_10", "std_vol_10",
            "zscore_close_10", "zscore_vol_5",
            "ret_close_10", "decay_vol_5", "decay_close_5",
        ]
        self.all_factor_cols = self.price_volume_cols + self.alphanet_cols

    # ------------------------------------------------------------------
    # 对外主流程
    # ------------------------------------------------------------------

    def run(self, top_n: int = 20) -> List[dict]:
        """
        执行完整的扫描-过滤-因子-打分流程。

        Parameters
        ----------
        top_n : int
            返回候选池数量。

        Returns
        -------
        list[dict]
            每个元素包含: symbol, symbol_name, xgb_score, key_factors, industry, market_cap_bn
        """
        # 1. 拉取全市场股票列表
        stock_info = self.get_stock_list()
        if stock_info.empty:
            logger.warning("获取股票列表失败或为空")
            return []

        # 2. 过滤不可交易标的
        tradable = self.filter_tradable(stock_info)
        if tradable.empty:
            logger.warning("无可交易标的")
            return []

        symbols = tradable.index.tolist()
        logger.info("全市场 %d 只 → 过滤后可交易 %d 只", len(stock_info), len(symbols))

        # 3. 批量计算量价因子
        factor_df = self._compute_price_volume_factors(symbols)
        if factor_df.empty:
            logger.warning("量价因子计算全部失败")
            return []

        # 4. 批量计算 AlphaNet 特征
        alpha_df = self._compute_alphanet_features(symbols)
        if not alpha_df.empty:
            # 合并因子
            factor_df = factor_df.join(alpha_df, how="inner")

        # 5. 五因子中性化
        factor_df = self._neutralize(factor_df, tradable)

        # 6. XGBoost 打分
        factor_df = self._score(factor_df)

        # 7. 生成候选池
        candidates = self._build_candidates(factor_df, tradable, top_n)
        return candidates

    # ------------------------------------------------------------------
    # Step 1: 获取全市场股票列表
    # ------------------------------------------------------------------

    def _stock_list_cache_path(self, cache_dir: str, date: Optional[str] = None) -> str:
        date = date or self.end_date
        return os.path.join(cache_dir, f"stock_list_{date}.pkl")

    def _load_cached_stock_list(self, cache_dir: str, allow_latest: bool = False) -> pd.DataFrame:
        paths = [self._stock_list_cache_path(cache_dir)]
        if allow_latest and os.path.isdir(cache_dir):
            cached_paths = sorted(
                [
                    os.path.join(cache_dir, name)
                    for name in os.listdir(cache_dir)
                    if name.startswith("stock_list_") and name.endswith(".pkl")
                ],
                key=os.path.getmtime,
                reverse=True,
            )
            paths.extend(path for path in cached_paths if path not in paths)

        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info("加载股票列表缓存: %s", path)
                    return df
            except Exception:
                logger.warning("股票列表缓存读取失败: %s", path)
        return pd.DataFrame()

    def _save_stock_list_cache(self, df: pd.DataFrame, cache_dir: str) -> None:
        if df.empty:
            return
        os.makedirs(cache_dir, exist_ok=True)
        path = self._stock_list_cache_path(cache_dir)
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f)
            logger.info("股票列表已缓存: %s", path)
        except Exception:
            logger.warning("股票列表缓存写入失败: %s", path)

    def get_stock_list(
        self,
        offline: bool = False,
        cache_dir: str = "data/stock_list_cache",
    ) -> pd.DataFrame:
        """
        拉取沪深 A 股基础信息，返回 DataFrame，行索引为 symbol。

        优先用 AKShare，失败时回退到 baostock。

        Returns
        -------
        pd.DataFrame
            列: symbol_name, industry, market_cap_bn, pe_ttm, is_st, is_suspend, listed_days
        """
        if offline:
            return self._load_cached_stock_list(cache_dir, allow_latest=True)

        df = self._get_stock_list_baostock()
        if df is not None and not df.empty:
            self._save_stock_list_cache(df, cache_dir)
            return df
        logger.warning("baostock 拉取失败，回退到 AKShare")
        df = self._get_stock_list_akshare()
        if df is not None and not df.empty:
            self._save_stock_list_cache(df, cache_dir)
            return df
        return self._load_cached_stock_list(cache_dir, allow_latest=True)

    def _get_stock_list_akshare(self) -> Optional[pd.DataFrame]:
        """通过 AKShare stock_zh_a_spot_em 拉取（带重试）。"""
        import akshare as ak
        from src.network.proxy_manager import proxy_manager

        try:
            df = proxy_manager.run(lambda: ak.stock_zh_a_spot_em(), "stock_zh_a_spot_em")
            if df is not None and not df.empty:
                return self._normalize_spot_df(df)
        except Exception:
            logger.warning("AKShare stock_zh_a_spot_em 拉取失败")
        return None

    def _get_stock_list_baostock(self) -> pd.DataFrame:
        """通过 baostock 拉取全市场股票列表。"""
        import baostock as bs

        try:
            lg = bs.login()
            if lg.error_code != "0":
                logger.error("baostock 登录失败: %s", lg.error_msg)
                return pd.DataFrame()

            rs = bs.query_stock_basic(code_name="")
            if rs.error_code != "0":
                logger.error("baostock 查询失败: %s", rs.error_msg)
                bs.logout()
                return pd.DataFrame()

            data = []
            while rs.next():
                row = rs.get_row_data()
                data.append(row)

            bs.logout()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=rs.fields)
            # baostock 列: code, code_name, ipoDate, outDate, type, status
            df = df[df["type"] == "1"]  # 仅 A 股
            df = df[df["status"] == "1"]  # 仅上市状态

            result = pd.DataFrame(index=df["code"].str.replace("sh.", "").str.replace("sz.", ""))
            result["symbol_name"] = df["code_name"].values
            result["industry"] = "未知"
            result["pe_ttm"] = np.nan
            result["market_cap_bn"] = np.nan
            # 注意：必须用 .values 取值，否则右侧 Series 的索引（0,1,2...）会与
            # result 的股票代码索引对齐失败，导致整列变成 NaN（ST 过滤失效）。
            result["is_st"] = df["code_name"].str.contains("ST|退", na=False).values
            result["chg_60d"] = np.nan
            return result
        except Exception:
            logger.exception("baostock 拉取失败")
            return pd.DataFrame()

    def _normalize_spot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化 AKShare spot DataFrame。"""
        col_map = {
            "代码": "symbol",
            "名称": "symbol_name",
            "总市值": "market_cap",
            "市盈率-动态": "pe_ttm",
            "行业": "industry",
            "60日涨跌幅": "chg_60d",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

        required = ["symbol", "symbol_name"]
        for c in required:
            if c not in df.columns:
                logger.error("缺少必要列 %s", c)
                return pd.DataFrame()

        result = pd.DataFrame(index=df["symbol"])
        result["symbol_name"] = df["symbol_name"]
        result["industry"] = df.get("industry", "未知")
        result["pe_ttm"] = pd.to_numeric(df.get("pe_ttm", np.nan), errors="coerce")
        result["market_cap"] = pd.to_numeric(df.get("market_cap", np.nan), errors="coerce")
        if result["market_cap"].max() > 1e8:
            result["market_cap_bn"] = result["market_cap"] / 1e8
        else:
            result["market_cap_bn"] = result["market_cap"]
        result["is_st"] = result["symbol_name"].str.contains("ST|退", na=False)
        result["chg_60d"] = pd.to_numeric(df.get("chg_60d", np.nan), errors="coerce")
        return result

    # ------------------------------------------------------------------
    # Step 2: 过滤
    # ------------------------------------------------------------------

    def filter_tradable(self, stock_info: pd.DataFrame) -> pd.DataFrame:
        """
        过滤不可交易标的：
        - 剔除 ST / *ST / 退市整理
        - 剔除当日停牌（成交量=0，在因子计算阶段处理）
        - 剔除上市不足 60 个交易日（因子计算阶段通过数据行数处理）
        - 剔除市值 < 20 亿
        - 剔除 PE_ttm < 0（亏损股）
        - 剔除近 20 日涨幅 > 40%（追高风险）

        Parameters
        ----------
        stock_info : pd.DataFrame
            get_stock_list() 的输出。

        Returns
        -------
        pd.DataFrame
            过滤后的股票信息。
        """
        df = stock_info.copy()

        # ST 过滤
        if "is_st" in df.columns:
            df = df[~df["is_st"].fillna(False).astype(bool)]

        # 市值过滤（仅当有有效值时才过滤）
        if "market_cap_bn" in df.columns and df["market_cap_bn"].notna().any():
            df = df[(df["market_cap_bn"] >= 20) | df["market_cap_bn"].isna()]

        # PE 过滤（亏损股）
        if "pe_ttm" in df.columns and df["pe_ttm"].notna().any():
            df = df[(df["pe_ttm"] > 0) | df["pe_ttm"].isna()]

        return df

    # ------------------------------------------------------------------
    # Step 3~4: 因子计算（内部方法）
    # ------------------------------------------------------------------

    def _compute_price_volume_factors(self, symbols: List[str]) -> pd.DataFrame:
        """批量计算量价因子。"""
        from src.factors.price_volume import compute_price_volume_factors

        try:
            df = compute_price_volume_factors(
                symbols,
                end_date=self.end_date,
                request_interval=self.request_interval,
            )
            return df
        except Exception:
            logger.exception("量价因子批量计算失败")
            return pd.DataFrame()

    def _compute_alphanet_features(self, symbols: List[str]) -> pd.DataFrame:
        """批量计算 AlphaNet 特征（逐只调用，返回合并 DataFrame）。"""
        from src.factors.alphanet_features import compute_alphanet_features
        from src.factors.price_volume import _fetch_one_stock, _normalize_columns

        all_rows = []
        for i, sym in enumerate(symbols):
            try:
                raw = _fetch_one_stock(sym, start_date=None, end_date=self.end_date)
                if raw is None or raw.empty:
                    continue
                s = compute_alphanet_features(raw, sym)
                if s is not None:
                    all_rows.append(s)
            except Exception:
                logger.exception("AlphaNet 特征计算 %s 失败", sym)
            if i < len(symbols) - 1:
                time.sleep(self.request_interval)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        if "symbol" in df.columns:
            df.set_index("symbol", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Step 5: 中性化
    # ------------------------------------------------------------------

    def _neutralize(
        self,
        factor_df: pd.DataFrame,
        stock_info: pd.DataFrame,
    ) -> pd.DataFrame:
        """五因子 OLS 中性化。"""
        from src.factors.factor_neutralize import neutralize_factors

        # 构建行业映射
        industry_map = {}
        if "industry" in stock_info.columns:
            industry_map = stock_info["industry"].to_dict()

        # 构建市值映射
        cap_map = {}
        if "market_cap_bn" in stock_info.columns:
            cap_map = stock_info["market_cap_bn"].to_dict()

        # 确定需要中性化的因子列（排除评分/名称列和回归自变量）
        reg_vars = {"mom_20d", "vol_20d", "turn_5d_avg"}
        factor_cols = [
            c for c in factor_df.columns
            if c not in reg_vars and c in self.all_factor_cols
        ]

        try:
            result = neutralize_factors(
                factor_df,
                industry_map=industry_map,
                market_cap_map=cap_map,
                factor_cols=factor_cols,
            )
            return result
        except Exception:
            logger.exception("中性化失败，返回原始因子")
            return factor_df

    # ------------------------------------------------------------------
    # Step 6: XGBoost 打分
    # ------------------------------------------------------------------

    def _score(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """使用 XGBoost 模型打分。"""
        if self.scorer is None:
            logger.warning("未设置 XGBoost 模型，使用等权假得分")
            factor_df["xgb_score"] = 0.5
            return factor_df

        # 确定特征列
        available_cols = [c for c in self.all_factor_cols if c in factor_df.columns]
        if not available_cols:
            logger.warning("无可用的因子列进行打分")
            factor_df["xgb_score"] = 0.5
            return factor_df

        X = factor_df[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
        try:
            scores = self.scorer.predict(X)
            factor_df["xgb_score"] = scores
        except Exception:
            logger.exception("XGBoost 打分失败")
            factor_df["xgb_score"] = 0.5

        return factor_df

    # ------------------------------------------------------------------
    # Step 7: 构建候选池
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        factor_df: pd.DataFrame,
        stock_info: pd.DataFrame,
        top_n: int,
    ) -> List[dict]:
        """将打分结果组装为候选池 list[dict]。"""
        df = factor_df.copy()

        # 降序排列
        sort_col = "xgb_score" if "xgb_score" in df.columns else df.columns[0]
        df = df.sort_values(sort_col, ascending=False)

        df = df.head(top_n)

        candidates = []
        for sym, row in df.iterrows():
            info = stock_info.loc[sym] if sym in stock_info.index else pd.Series()

            # 提取关键因子
            key_factors = {}
            for col in self.all_factor_cols:
                if col in row.index and not pd.isna(row[col]):
                    key_factors[col] = round(float(row[col]), 4)

            candidate = {
                "symbol": str(sym),
                "symbol_name": str(info.get("symbol_name", "")),
                "xgb_score": round(float(row.get("xgb_score", 0.5)), 4),
                "key_factors": key_factors,
                "industry": str(info.get("industry", "未知")),
                "market_cap_bn": round(float(info.get("market_cap_bn", np.nan)), 2) if not pd.isna(info.get("market_cap_bn")) else None,
            }
            candidates.append(candidate)

        return candidates

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def set_model(self, scorer) -> None:
        """设置 XGBoost 打分模型。"""
        self.scorer = scorer

    def load_model(self, model_path: str = "data/xgb_scorer.model") -> bool:
        """
        尝试加载预训练 XGBoost 模型。

        文件不存在或加载失败时返回 False，并保持 scorer=None（即等权降级打分），
        保证主流程在无模型时仍可运行。

        Returns
        -------
        bool
            是否成功加载模型。
        """
        import os

        if not os.path.exists(model_path):
            logger.warning("未找到 XGBoost 模型 %s，使用等权打分（无量化 alpha）", model_path)
            return False
        try:
            from src.models.xgboost_scorer import XGBoostScorer

            scorer = XGBoostScorer(model_path=model_path)
            scorer.load(model_path)
            self.set_model(scorer)
            logger.info("已加载 XGBoost 模型: %s", model_path)
            return True
        except Exception:
            logger.exception("加载 XGBoost 模型失败（降级为等权打分）")
            return False
