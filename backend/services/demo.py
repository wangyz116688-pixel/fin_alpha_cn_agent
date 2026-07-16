"""Local demo-chat service backed by the official backtest artifacts."""

from __future__ import annotations

import json
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOGS = ROOT / "logs"
DEMO_CACHE = LOGS / "demo_daily"
INITIAL_CAPITAL = 500_000.0

STRATEGY = {
    "name": "AlphaAgent IC Blend",
    "score_method": "train_ic_blend",
    "allocation_method": "score",
    "position_count": 3,
    "stock_pool_size": 500,
    "random_seed": 42,
    "initial_capital": INITIAL_CAPITAL,
    "max_drawdown_stop": 0,
}

SOURCES = [
    {
        "key": "2024",
        "label": "2024",
        "advice": LOGS / "full_2024_default_final_s500" / "advice_20240101_20241231_s500.json",
        "equity": LOGS / "full_2024_default_final_s500" / "backtest_20240101_20241231_s500.csv",
    },
    {
        "key": "2025",
        "label": "2025",
        "advice": LOGS / "full_2025_default_final_s500" / "advice_20250101_20251231_s500.json",
        "equity": LOGS / "full_2025_default_final_s500" / "backtest_20250101_20251231_s500.csv",
    },
    {
        "key": "2026_h1",
        "label": "2026上半年",
        "advice": LOGS / "h1_2026_default_final_s500" / "advice_20260101_20260630_s500.json",
        "equity": LOGS / "h1_2026_default_final_s500" / "backtest_20260101_20260630_s500.csv",
    },
    {
        "key": "2026_recent",
        "label": "2026近期正式策略",
        "advice": LOGS / "daily_final_advice" / "advice_20260710_20260713_s500.json",
        "equity": LOGS / "daily_final_advice" / "backtest_20260710_20260713_s500.csv",
    },
]

FACTOR_LABELS = {
    "turn_5d_avg": "近5日换手率",
    "mom_20d": "20日动量",
    "vol_20d": "20日波动率",
    "vwap_dev": "成交均价偏离",
    "zscore_vol_5": "5日量能异常度",
    "turn_ratio": "换手率变化",
    "rev_1d": "短期反转",
    "std_vol_10": "10日成交量离散度",
    "std_close_10": "10日价格离散度",
    "decay_vol_5": "衰减成交量",
}

WEIGHTS = {
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

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-chat")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@lru_cache(maxsize=1)
def _advice_index() -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], list[str]]:
    index: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for source in SOURCES:
        for raw_date, positions in _read_json(source["advice"]).items():
            iso = datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")
            index[iso] = (source, positions)
    DEMO_CACHE.mkdir(parents=True, exist_ok=True)
    for path in DEMO_CACHE.glob("*.json"):
        try:
            result = _read_json(path)
            if result.get("date"):
                index[result["date"]] = ({"key": "demo", "label": "实时计算"}, result.get("raw_positions", []))
        except (OSError, ValueError):
            continue
    return index, sorted(index)


def _parse_requested_date(message: str, explicit: Optional[str] = None) -> Optional[date]:
    raw = (explicit or "").strip()
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("日期格式应为 YYYY-MM-DD。") from exc
    text = message.strip()
    if "今天" in text:
        return date.today()
    match = re.search(r"(20\d{2})[年/\-.](\d{1,2})[月/\-.](\d{1,2})日?", text)
    if match:
        try:
            return date(*map(int, match.groups()))
        except ValueError as exc:
            raise ValueError("消息中的日期无效，请检查年月日。") from exc
    match = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", text)
    if match:
        try:
            return date(*map(int, match.groups()))
        except ValueError as exc:
            raise ValueError("消息中的日期无效，请检查年月日。") from exc
    match = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})日", text)
    if match:
        try:
            return date(date.today().year, int(match.group(1)), int(match.group(2)))
        except ValueError as exc:
            raise ValueError("消息中的日期无效，请检查月日。") from exc
    return None


def _intent(message: str, requested: Optional[date]) -> str:
    if any(word in message for word in ("指标", "收益", "回撤", "夏普", "卡玛", "索泰诺", "表现", "净值")):
        return "metrics"
    if any(word in message for word in ("策略说明", "策略逻辑", "什么策略", "因子", "原理")):
        return "strategy"
    if requested or any(word in message for word in ("建议", "选股", "仓位", "买什么", "今天", "最近交易日")):
        return "advice"
    return "help"


@lru_cache(maxsize=8)
def _equity_frame(key: str) -> pd.DataFrame:
    source = next(item for item in SOURCES if item["key"] == key)
    frame = pd.read_csv(source["equity"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _capital_before(source: dict[str, Any], trade_date: str) -> float:
    if source.get("key") not in {item["key"] for item in SOURCES}:
        return INITIAL_CAPITAL
    frame = _equity_frame(source["key"])
    prior = frame.loc[frame["date"] < pd.Timestamp(trade_date), "total_assets"]
    return float(prior.iloc[-1]) if len(prior) else INITIAL_CAPITAL


@lru_cache(maxsize=2048)
def _previous_close(symbol: str, trade_date: str) -> Optional[float]:
    target = pd.Timestamp(trade_date)
    candidates = []
    for folder in (ROOT / "data" / "prefetch_cache", ROOT / "data" / "train_cache"):
        candidates.extend(folder.glob(f"{symbol}_*.pkl"))
    for path in sorted(candidates, key=lambda p: p.stat().st_size, reverse=True):
        try:
            frame = pd.read_pickle(path)
            frame["date"] = pd.to_datetime(frame["date"])
            prior = frame.loc[frame["date"] < target]
            if not prior.empty:
                return float(prior.iloc[-1]["close"])
        except (OSError, KeyError, ValueError):
            continue
    return None


def _factor_details(symbols: list[str], as_of_date: str) -> dict[str, dict[str, Any]]:
    target = as_of_date.replace("-", "")
    files = list((ROOT / "data" / "factor_cache").glob(f"factors_v2_{target}_*500*.pkl"))
    files += list((ROOT / "data" / "factor_cache_train").glob(f"factors_v2_{target}_*500*.pkl"))
    best: Optional[pd.DataFrame] = None
    best_matches = -1
    for path in files:
        try:
            frame = pd.read_pickle(path)
            frame.index = frame.index.astype(str).str.zfill(6)
            matches = len(set(symbols) & set(frame.index))
            if matches > best_matches:
                best, best_matches = frame, matches
        except (OSError, ValueError):
            continue
    if best is None or best_matches <= 0:
        return {}
    available = [name for name in WEIGHTS if name in best.columns]
    oriented = pd.DataFrame(index=best.index)
    for name in available:
        series = best[name] if WEIGHTS[name] > 0 else -best[name]
        oriented[name] = series.rank(pct=True)
    abs_weights = np.array([abs(WEIGHTS[name]) for name in available])
    score = np.average(oriented.values, axis=1, weights=abs_weights)
    details: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        if symbol not in best.index:
            continue
        row = oriented.loc[symbol].sort_values(ascending=False)
        top = []
        for name in row.index[:3]:
            top.append({"name": FACTOR_LABELS.get(name, name), "percentile": round(float(row[name]), 4)})
        details[symbol] = {"score": round(float(score[best.index.get_loc(symbol)]), 4), "factors": top}
    return details


def _build_recommendation(requested: date, resolved: str, source: dict[str, Any], raw_positions: list[dict[str, Any]], origin: str = "cache") -> dict[str, Any]:
    as_of = (datetime.strptime(resolved, "%Y-%m-%d") - timedelta(days=1)).date()
    # Factor files use the immediately preceding trading day, not always calendar day - search backwards.
    factor_info: dict[str, dict[str, Any]] = {}
    symbols = [str(item["symbol"]).zfill(6) for item in raw_positions]
    for offset in range(1, 8):
        candidate = (datetime.strptime(resolved, "%Y-%m-%d").date() - timedelta(days=offset)).isoformat()
        factor_info = _factor_details(symbols, candidate)
        if factor_info:
            as_of = datetime.strptime(candidate, "%Y-%m-%d").date()
            break
    capital = _capital_before(source, resolved)
    positions = []
    invested = 0.0
    for item in raw_positions:
        symbol = str(item["symbol"]).zfill(6)
        supplied_close = item.get("previous_close")
        close = float(supplied_close) if supplied_close is not None else _previous_close(symbol, resolved)
        supplied_amount = item.get("amount")
        amount = float(supplied_amount) if supplied_amount is not None else (float(item["volume"]) * close if close is not None else None)
        invested += amount or 0.0
        info = factor_info.get(symbol, {})
        score = item.get("score", info.get("score"))
        factors = info.get("factors", [])
        reason = "、".join(f"{factor['name']}处于市场前{max(1, round((1-factor['percentile'])*100))}%" for factor in factors[:2])
        if not reason:
            reason = "train_ic_blend 综合评分进入当日 Top 3"
        positions.append({
            "symbol": symbol,
            "name": item.get("symbol_name") or symbol,
            "shares": int(item["volume"]),
            "previous_close": round(close, 4) if close is not None else None,
            "amount": round(amount, 2) if amount is not None else None,
            "weight": round(amount / capital, 6) if amount is not None and capital else None,
            "score": score,
            "confidence": score,
            "factors": factors,
            "reason": reason,
        })
    adjusted = requested.isoformat() != resolved
    return {
        "date": resolved,
        "requested_date": requested.isoformat(),
        "signal_date": as_of.isoformat(),
        "adjusted": adjusted,
        "adjustment_note": f"{requested.isoformat()}非可用交易日，已回退至{resolved}。" if adjusted else None,
        "source": origin,
        "source_label": "历史回测缓存" if origin == "cache" else "本次实时计算",
        "strategy": STRATEGY,
        "available_capital": round(capital, 2),
        "invested_amount": round(invested, 2),
        "cash": round(max(0.0, capital - invested), 2),
        "positions": positions,
        "risk_notice": "本建议由历史数据和量化模型生成，仅用于研究演示，不构成任何投资建议。",
    }


def get_overview() -> dict[str, Any]:
    metrics_path = LOGS / "factor_analysis" / "final_performance_metrics.csv"
    frame = pd.read_csv(metrics_path)
    metrics = []
    key_map = {"2024 full year": "2024", "2025 full year": "2025", "2026 H1": "2026_h1"}
    label_map = {"2024 full year": "2024", "2025 full year": "2025", "2026 H1": "2026上半年"}
    for _, row in frame.iterrows():
        metrics.append({
            "key": key_map[row["period"]], "label": label_map[row["period"]], "days": int(row["days"]),
            "total_return": float(row["total_return"]), "annual_return": float(row["annual_return"]),
            "max_drawdown": float(row["max_drawdown"]), "sharpe": float(row["sharpe"]),
            "sortino": float(row["sortino"]), "calmar": float(row["calmar"]),
            "win_rate": float(row["win_rate"]), "profit_loss_ratio": float(row["profit_loss_ratio"]),
        })
    equity = _equity_frame("2026_h1")
    values = [INITIAL_CAPITAL] + equity["total_assets"].astype(float).tolist()
    dates = ["2026-01-01"] + equity["date"].dt.strftime("%Y-%m-%d").tolist()
    curve = [{"date": d, "value": v, "return": v / INITIAL_CAPITAL - 1} for d, v in zip(dates, values)]
    _, available = _advice_index()
    return {
        "strategy": STRATEGY,
        "available_range": {"start": available[0], "end": available[-1]},
        "metrics": metrics,
        "default_period": "2026_h1",
        "equity_curve": curve,
        "quick_questions": ["今天交易建议", "2026-06-30 的建议", "查看策略表现"],
    }


def _job_update(job_id: str, **values: Any) -> None:
    with _lock:
        _jobs[job_id].update(values)


def _latest_local_universe(requested: date) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.DataFrame]:
    """Load the latest complete 500-stock snapshot strictly before trade date."""
    groups: dict[tuple[str, str], list[Path]] = {}
    cutoff = requested.strftime("%Y%m%d")
    pattern = re.compile(r"^(\d{6})_(\d{8})_(\d{8})\.pkl$")
    for path in (ROOT / "data" / "prefetch_cache").glob("*.pkl"):
        match = pattern.match(path.name)
        if match and match.group(3) < cutoff:
            groups.setdefault((match.group(2), match.group(3)), []).append(path)
    eligible = [(key, paths) for key, paths in groups.items() if len(paths) >= 450]
    if not eligible:
        raise RuntimeError("本地没有覆盖信号日的完整500只股票池行情。")
    (start, end), paths = max(eligible, key=lambda item: (item[0][1], -abs(len(item[1]) - 500)))
    # A formal universe is exactly 500 names. Ignore unrelated files if a folder contains more.
    paths = sorted(paths)[:500]
    price_map: dict[str, pd.DataFrame] = {}
    for path in paths:
        symbol = path.name[:6]
        frame = pd.read_pickle(path)
        frame["date"] = pd.to_datetime(frame["date"])
        price_map[symbol] = frame
    dates = sorted({stamp for frame in price_map.values() for stamp in frame.loc[frame["date"] < pd.Timestamp(requested), "date"]})
    if not dates:
        raise RuntimeError("本地行情中没有早于建议日的交易信号。")
    as_of = dates[-1]
    stock_files = sorted((ROOT / "data" / "stock_list_cache").glob("stock_list_*.pkl"), reverse=True)
    stock_info = pd.DataFrame()
    preferred = ROOT / "data" / "stock_list_cache" / f"stock_list_{requested.strftime('%Y%m%d')}.pkl"
    for path in ([preferred] if preferred.exists() else []) + stock_files:
        try:
            candidate = pd.read_pickle(path)
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                candidate.index = candidate.index.astype(str).str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False).str.zfill(6)
                stock_info = candidate.loc[candidate.index.intersection(price_map.keys())].copy()
                if len(stock_info) >= 450:
                    break
        except (OSError, ValueError):
            continue
    if stock_info.empty:
        stock_info = pd.DataFrame(index=pd.Index(price_map.keys(), name="symbol"))
        stock_info["symbol_name"] = stock_info.index
    return price_map, as_of, stock_info


def _generate_from_local_signal(requested: date) -> tuple[str, list[dict[str, Any]]]:
    """Generate next-session advice from the prior session; no trade-day close is needed."""
    from src.backtest_stock_agent import _build_decisions_for_method
    from src.execution.position_sizer import PositionSizer
    from src.models.stock_screener import StockScreener

    price_map, as_of, stock_info = _latest_local_universe(requested)
    screener = StockScreener(end_date=requested.strftime("%Y%m%d"), request_interval=0)
    screener.load_model(str(ROOT / "data" / "xgb_scorer.model"))
    decisions = _build_decisions_for_method(
        screener=screener, price_map=price_map, as_of_date=as_of, stock_info=stock_info,
        top_n=20, factor_cache_dir=str(ROOT / "data" / "factor_cache"),
        score_method="train_ic_blend", available_capital=INITIAL_CAPITAL,
        position_count=3, confidence_threshold=0.55, fill_positions=True,
        allocation_method="score", sizer=PositionSizer(allocation_method="score"),
    )
    if not decisions:
        raise RuntimeError("本地信号计算完成，但正式策略未选出可交易持仓。")
    raw = []
    for item in decisions:
        previous_close = float(item.get("prev_close", 0))
        volume = int(item["volume"])
        raw.append({
            "symbol": str(item["symbol"]).zfill(6),
            "symbol_name": item.get("symbol_name") or str(item["symbol"]).zfill(6),
            "volume": volume,
            "previous_close": previous_close,
            "amount": round(previous_close * volume, 2),
            "score": float(item.get("C_mixed", item.get("xgb_score", 0.5))),
        })
    return as_of.strftime("%Y-%m-%d"), raw


def _run_live_job(job_id: str, requested: date) -> None:
    try:
        _job_update(job_id, stage="准备股票池", progress=12)
        try:
            _job_update(job_id, stage="读取最近交易日行情", progress=28)
            signal_date, raw = _generate_from_local_signal(requested)
            _job_update(job_id, stage="排序配资并生成建议", progress=88)
            payload = _build_recommendation(requested, requested.isoformat(), {"key": "demo", "label": "实时计算"}, raw, "live")
            payload["signal_date"] = signal_date
            payload["raw_positions"] = raw
            DEMO_CACHE.mkdir(parents=True, exist_ok=True)
            (DEMO_CACHE / f"{requested.isoformat()}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _advice_index.cache_clear()
            _job_update(job_id, status="completed", stage="生成建议", progress=100, result=payload)
            return
        except RuntimeError as local_error:
            _job_update(job_id, stage="本地缓存不足，尝试补充行情", progress=35, detail=str(local_error))

        from src.backtest_stock_agent import run_backtest

        run_dir = DEMO_CACHE / "runs" / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        start = requested - timedelta(days=10)
        _job_update(job_id, stage="加载行情并计算因子", progress=35)
        result = run_backtest(
            start_date=start.strftime("%Y%m%d"), end_date=requested.strftime("%Y%m%d"),
            initial_capital=INITIAL_CAPITAL, max_stocks=500, cache_dir=str(ROOT / "data" / "prefetch_cache"),
            log_dir=str(run_dir), factor_cache_dir=str(ROOT / "data" / "factor_cache"), no_plot=True,
            score_method="train_ic_blend", position_count=3, allocation_method="score",
            max_drawdown_stop=0, print_advice=False,
        )
        _job_update(job_id, stage="排序配资并生成建议", progress=88)
        advice_files = sorted(run_dir.glob("advice_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if result.empty or not advice_files:
            raise RuntimeError("未获得可用交易日或行情数据。")
        advice = _read_json(advice_files[0])
        raw = advice.get(requested.strftime("%Y%m%d"))
        if raw is None:
            last_key = sorted(advice)[-1]
            resolved = datetime.strptime(last_key, "%Y%m%d").strftime("%Y-%m-%d")
            raw = advice[last_key]
        else:
            resolved = requested.isoformat()
        payload = _build_recommendation(requested, resolved, {"key": "demo", "label": "实时计算"}, raw, "live")
        payload["raw_positions"] = raw
        DEMO_CACHE.mkdir(parents=True, exist_ok=True)
        (DEMO_CACHE / f"{resolved}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _advice_index.cache_clear()
        _job_update(job_id, status="completed", stage="生成建议", progress=100, result=payload)
    except Exception as exc:  # background boundary: return readable error to UI
        _job_update(job_id, status="failed", stage="计算失败", progress=100, error=str(exc))


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def chat(message: str, explicit_date: Optional[str] = None) -> dict[str, Any]:
    requested = _parse_requested_date(message, explicit_date)
    intent = _intent(message, requested)
    if intent == "metrics":
        return {"status": "completed", "intent": "metrics", "reply": "以下为正式策略的分区间表现，默认展示2026上半年。", "overview": get_overview()}
    if intent == "strategy":
        return {"status": "completed", "intent": "strategy", "reply": "策略以固定种子500只股票池为基础，使用train_ic_blend多因子评分、按得分分配资金，每日持有3只股票。", "strategy": STRATEGY}
    if intent == "help":
        return {"status": "completed", "intent": "help", "reply": "你可以问“今天交易建议”“2026年6月30日建议”或“查看策略表现”。"}
    requested = requested or date.today()
    if requested > date.today():
        raise ValueError("未来日期尚无行情数据，请选择今天或更早的日期。")
    index, available = _advice_index()
    requested_iso = requested.isoformat()
    if requested_iso in index:
        source, raw = index[requested_iso]
        result = _build_recommendation(requested, requested_iso, source, raw, "live" if source.get("key") == "demo" else "cache")
        return {"status": "completed", "intent": "advice", "reply": f"{result['date']} 交易建议已生成，共{len(result['positions'])}只持仓。", "result": result}
    prior = [item for item in available if item <= requested_iso]
    if requested_iso <= available[-1] and prior:
        resolved = prior[-1]
        source, raw = index[resolved]
        result = _build_recommendation(requested, resolved, source, raw, "cache")
        return {"status": "completed", "intent": "advice", "reply": result["adjustment_note"], "result": result}
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"job_id": job_id, "status": "pending", "stage": "准备股票池", "progress": 5, "requested_date": requested_iso}
    _executor.submit(_run_live_job, job_id, requested)
    return {"status": "pending", "intent": "advice", "reply": "该日期没有缓存，已启动正式策略计算。", "job_id": job_id, "stage": "准备股票池", "progress": 5}
