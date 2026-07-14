"""
XGBoost 打分模型。

使用 `rank:pairwise` 目标训练排序模型，预测次日收益率截面排名。
支持初始训练、推理打分、滚动微调三种模式。

对外接口:
    XGBoostScorer 类
        - train(X, y, groups, eval_set)
        - predict(X) -> np.ndarray (scores)
        - fine_tune(X, y, groups, eval_set)
        - save(path) / load(path)
"""

import logging
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class XGBoostScorer:
    """
    XGBoost 排序打分模型。

    Parameters
    ----------
    params : dict, optional
        XGBoost 超参数。默认使用 PROJECT_SPEC 中的配置。
    model_path : str, optional
        模型保存/加载路径。
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        model_path: Optional[str] = None,
    ):
        import xgboost as xgb

        self.xgb = xgb
        self.model: Optional[xgb.Booster] = None
        self.model_path = model_path or "data/xgb_scorer.model"
        self.feature_names: Optional[List[str]] = None

        # 默认参数
        if params is None:
            params = {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "rank:pairwise",
                "eval_metric": "ndcg",
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            }
        self.params = params.copy()

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        eval_groups: Optional[List[np.ndarray]] = None,
        early_stopping_rounds: int = 20,
        verbose: bool = True,
    ) -> dict:
        """
        初始训练 XGBoost 排序模型。

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            训练特征矩阵。
        y : np.ndarray, shape (n_samples,)
            目标变量（次日收益率截面排名分档，0~9）。
        groups : np.ndarray, optional
            分组数组（用于 rank:pairwise），如每日截面分组。
        eval_set : list of (X_eval, y_eval), optional
            验证集。
        eval_groups : list of np.ndarray, optional
            验证集分组。
        early_stopping_rounds : int
            早停轮数。
        verbose : bool
            是否输出训练日志。

        Returns
        -------
        dict
            训练评估结果字典。
        """
        # 构建 DMatrix
        dtrain = self._make_dmatrix(X, y, groups)
        dval = None
        if eval_set:
            dval = [
                (
                    self._make_dmatrix(Xv, yv, eval_groups[i] if eval_groups else None),
                    f"eval_{i}",
                )
                for i, (Xv, yv) in enumerate(eval_set)
            ]

        # evals 必须是 [(DMatrix, name), ...] 元组列表
        evals = [(dtrain, "train")]
        if dval:
            evals.extend(dval)

        evals_result: dict = {}
        self.model = self.xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.params.get("n_estimators", 200),
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds if dval else None,
            verbose_eval=verbose,
        )

        # 保存特征名
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        logger.info(
            "训练完成，best_iteration=%s, best_score=%s",
            getattr(self.model, "best_iteration", None),
            getattr(self.model, "best_score", None),
        )
        return evals_result

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对特征矩阵打分，返回 0~1 归一化得分（得分越高越好）。

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            0~1 之间的预测得分。
        """
        if self.model is None:
            raise RuntimeError("模型未训练或未加载，请先调用 train() 或 load()")

        dmatrix = self._make_dmatrix(X, None, None)
        best_iteration = getattr(self.model, "best_iteration", None)
        iteration_range = None
        if best_iteration is not None:
            iteration_range = (0, int(best_iteration) + 1)

        try:
            raw = (
                self.model.predict(dmatrix, iteration_range=iteration_range)
                if iteration_range is not None
                else self.model.predict(dmatrix)
            )
        except TypeError:
            raw = self.model.predict(dmatrix)

        if raw.size == 0:
            return raw

        # 归一化到 [0, 1]
        raw_min = raw.min()
        raw_max = raw.max()
        if raw_max - raw_min > 1e-10:
            scores = (raw - raw_min) / (raw_max - raw_min)
        else:
            scores = np.zeros_like(raw)
        return scores

    # ------------------------------------------------------------------
    # 滚动微调
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        eval_groups: Optional[List[np.ndarray]] = None,
        num_boost_round: int = 10,
        early_stopping_rounds: int = 5,
        verbose: bool = False,
    ) -> dict:
        """
        滚动微调（增量训练）。在现有模型基础上继续训练若干轮。

        Parameters
        ----------
        X, y, groups, eval_set, eval_groups : 同 train()
        num_boost_round : int
            增量训练轮数。
        early_stopping_rounds : int
            早停轮数。
        verbose : bool

        Returns
        -------
        dict
            评估结果。
        """
        if self.model is None:
            raise RuntimeError("模型尚未初始化，请先 train() 或 load()")

        dtrain = self._make_dmatrix(X, y, groups)
        dval = None
        if eval_set:
            dval = [
                (
                    self._make_dmatrix(Xv, yv, eval_groups[i] if eval_groups else None),
                    f"eval_{i}",
                )
                for i, (Xv, yv) in enumerate(eval_set)
            ]

        evals_result: dict = {}
        self.model = self.xgb.train(
            params={**self.params, "process_type": "update",
                    "updater": "refresh", "refresh_leaf": False},
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            xgb_model=self.model,
            evals=[(dtrain, "train")] + (dval or []),
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds if dval else None,
            verbose_eval=verbose,
        )
        return evals_result

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """保存模型到文件。"""
        if self.model is None:
            raise RuntimeError("没有可保存的模型")
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "params": self.params,
                    "feature_names": self.feature_names,
                },
                f,
            )
        logger.info("模型已保存到 %s", save_path)

    def load(self, path: Optional[str] = None) -> None:
        """从文件加载模型。"""
        load_path = path or self.model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.params = data.get("params", self.params)
        self.feature_names = data.get("feature_names")
        logger.info("模型已从 %s 加载", load_path)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _make_dmatrix(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """构建 XGBoost DMatrix，支持分组排序。"""
        import xgboost as xgb

        kwargs = {"data": X}
        if y is not None:
            kwargs["label"] = y
        if groups is not None:
            kwargs["group"] = groups

        return xgb.DMatrix(**kwargs)

    # ------------------------------------------------------------------
    # 辅助：构建训练标签
    # ------------------------------------------------------------------

    @staticmethod
    def build_labels(
        returns: pd.Series,
        n_buckets: int = 10,
    ) -> pd.Series:
        """
        将每日截面收益率分档为 0~n_buckets-1 的标签。
        0 = 最差，n_buckets-1 = 最优。

        Parameters
        ----------
        returns : pd.Series
            次日收益率截面数据。
        n_buckets : int
            分档数。

        Returns
        -------
        pd.Series
            整数标签。
        """
        return returns.groupby(level=0).transform(
            lambda x: pd.qcut(x, n_buckets, labels=False, duplicates="drop")
            if len(x.dropna()) >= n_buckets
            else np.nan
        )
