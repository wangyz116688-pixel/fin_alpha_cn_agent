# 智投未来 · A 股日内投资智能体
## 项目设计说明文档

> **项目名称**：AlphaAgent —— 基于量价因子与多智能体协同的 A 股日内投资系统  
> **数据来源**：baostock（主，日频前复权 K 线）/ AKShare（备，财务、资金流、新闻）  
> **赛制**：初始资金 50 万元，日内交易，日终清仓，输出标准 JSON

> **关于本文档**：这是 AlphaAgent 的**完整系统设计文档**，描述目标形态与设计意图，其中第五章多智能体层、SHAP 分析、资金流因子等部分为规划/占位实现。**项目当前的实际落地状态以 [README.md](README.md) 第七章「当前实现状态」为准**，量化层（因子→中性化→XGBoost→选股→回测）已完整实现并可复现。

---

## 目录

1. [项目背景与设计思路](#第一章-项目背景与设计思路)
2. [系统架构](#第二章-系统架构)
3. [因子体系](#第三章-因子体系)
4. [量化选股层](#第四章-量化选股层)
5. [多智能体决策层](#第五章-多智能体决策层)
6. [资金管理与输出层](#第六章-资金管理与输出层)
7. [可解释性与日志设计](#第七章-可解释性与日志设计)
8. [开发路线图](#第八章-开发路线图)
9. [策略预期与风险说明](#第九章-策略预期与风险说明)

---

## 第一章 项目背景与设计思路

### 1.1 赛题目标解读

本赛道要求智能体每个交易日输出结构化投资建议 JSON，平台按**前一交易日收盘价买入**、**当日收盘价卖出**，日终自动清仓。核心评价维度：

- 累计总资产排名（初始 50 万元，日内单向盈亏积累）
- 策略可解释性：逻辑清晰、假设明确、可复现
- 稳健性：评测周期内的波动控制与连续盈利能力

### 1.2 核心设计哲学：量化选股 + 智能体决策

本系统采用**双层架构**，将量化因子的客观性与大语言模型的语义理解能力有机结合：

- **第一层（量化层）**：基于 A 股量价特征，用 XGBoost 模型对全市场股票打分，筛选出 Top20 候选标的。该层提供可量化、可回测的选股逻辑，结果稳定可复现。
- **第二层（智能体层）**：对候选标的调用多个专业 Agent 协同分析，涵盖技术面、基本面、情绪面、估值、宏观，并通过多空辩论机制输出带置信度的最终决策。该层提供完整的可解释性叙事。

两层架构解耦设计，各层可独立优化，也可根据市场状态快速调整参数。

### 1.3 系统核心特性

| 特性 | 说明 |
|------|------|
| 全市场自动扫描 | 每日从沪深两市约 5000 只股票中自动筛选，无需人工指定标的 |
| 量化因子体系 | 融合动量、波动、换手、量价相关等 20 个因子，参考学术主流量化框架 |
| 多智能体协同 | 10 个专业 Agent 分工协作，多空辩论机制提升决策客观性 |
| 标准化输出 | 直接输出赛制要求的 JSON 格式，volume 自动取 100 整数倍 |
| 完整决策日志 | 每日生成含推理链路的审计日志，满足「可审计与复现」要求 |

---

## 第二章 系统架构

### 2.1 五层流水线架构

```
每日 09:20 触发
        ↓
Layer 0  数据采集层     baostock（主）拉取全市场前复权 K 线；AKShare（备）资金流 / 新闻，缓存 SQLite
        ↓
Layer 1  因子计算层     量价因子(13) + AlphaNet 特征(10)（资金流因子已实现，暂未接入）
        ↓
Layer 2  量化选股层     五因子中性化 → XGBoost 打分 → Top-N 候选池
        ↓
Layer 3  智能体决策层   多 Agent 并行分析 → 混合置信度 C_mixed → 确认 3~5 只
        ↓
Layer 4  输出执行层    资金加权分配 → volume 取整 → 输出 JSON + 审计日志
```

### 2.2 项目目录结构

```
AlphaAgent/
├── src/
│   ├── agents/                        # 多智能体模块
│   │   ├── market_data.py             # 市场数据分析师（数据预处理入口）
│   │   ├── technicals.py              # 技术面分析师
│   │   ├── fundamentals.py            # 基本面分析师
│   │   ├── sentiment.py               # 情绪分析师
│   │   ├── valuation.py               # 估值分析师
│   │   ├── macro_analyst.py           # 宏观分析师
│   │   ├── researcher_bull.py         # 多方研究员
│   │   ├── researcher_bear.py         # 空方研究员
│   │   ├── debate_room.py             # 辩论室（LLM 第三方裁判）
│   │   ├── risk_manager.py            # 风险管理师（候选池排序）
│   │   └── portfolio_manager.py       # 投资组合管理师（最终决策）
│   ├── factors/                       # 量化因子体系
│   │   ├── price_volume.py            # 量价因子（动量/波动/换手/量价相关）
│   │   ├── alphanet_features.py       # 时序特征算子（6 类）
│   │   ├── capital_flow.py            # 资金流因子
│   │   └── factor_neutralize.py       # 五因子 OLS 中性化
│   ├── models/                        # 量化模型
│   │   ├── xgboost_scorer.py          # XGBoost 训练、推理、滚动微调
│   │   ├── train_xgboost.py           # 模型训练脚本（构建训练集 + 时序切分）
│   │   └── stock_screener.py          # 全市场扫描 → Top-N 候选池
│   ├── execution/                     # 输出执行层
│   │   ├── position_sizer.py          # 资金加权分配，volume 取整
│   │   └── output_formatter.py        # 输出 JSON + reasoning.md
│   ├── network/
│   │   └── proxy_manager.py           # 代理轮询 + AKShare no-proxy 补丁
│   ├── tools/                         # 数据接入（baostock_client / akshare_cache 等）
│   ├── main_stock_agent.py            # ★ 竞赛主入口（每日选股）
│   ├── backtest_stock_agent.py        # ★ 竞赛口径回测
│   ├── main.py                        # 原项目入口（单股票 LLM 分析，保留）
│   └── backtester.py                  # 原项目回测（单股票，保留）
├── data/                              # 本地缓存（K 线 / 训练集）+ xgb_scorer.model
├── logs/                              # 回测结果、每日建议 JSON、reasoning.md
├── PROJECT_SPEC.md                    # 本文档
├── 因子与选股策略说明.md               # 因子与选股可解释性说明
├── README.md                          # 使用指南与实现状态
└── .env                               # API Keys 配置（智能体层可选）
```

---

## 第三章 因子体系

### 3.1 因子分类总览

系统构建了覆盖量价、资金流、基本面三大类共约 20 个因子。参考量化金融学术研究成果，量价类因子在短周期选股中信息含量最高，因此权重最大。

| 类别 | 代表因子 | 作用 |
|------|---------|------|
| 量价类（核心） | 动量/波动/换手/量价相关 | 主要打分依据，信息含量约占 60% |
| 时序特征（AlphaNet 算子） | ts_corr / ts_zscore / ts_return | 非线性量价组合，提供增量 alpha |
| 资金流类 | 主力净流入/北向/龙虎榜 | 短期择时辅助信号 |
| 基本面过滤 | PE/市值/亏损标志 | 用于剔除低质量标的，不直接打分 |

### 3.2 量价因子（`factors/price_volume.py`）

使用 baostock 日频前复权 K 线（`adjust='qfq'`，AKShare 为备用源）：

**动量因子**
```python
mom_1d  = close[t-1] / close[t-2] - 1          # 1 日动量
mom_5d  = close[t-1] / close[t-6] - 1          # 5 日动量
mom_20d = close[t-1] / close[t-21] - 1         # 20 日动量（中性化基准）
rev_1d  = -mom_1d                               # 短期反转
```

**波动率因子**
```python
vol_5d    = rolling_std(daily_returns, 5)       # 5 日波动率
vol_20d   = rolling_std(daily_returns, 20)      # 20 日波动率（中性化基准）
vol_ratio = vol_5d / vol_20d                    # 波动率加速信号
```

**换手率因子**
```python
turn_1d     = 当日换手率
turn_5d_avg = rolling_mean(turn_1d, 5)          # 5 日均换手（中性化基准）
turn_ratio  = turn_1d / turn_5d_avg             # 相对换手（放量/缩量）
```

**量价关系因子**
```python
price_vol_corr_5  = corr(close[-5:], volume[-5:])   # 量价配合度
price_turn_corr_5 = corr(close[-5:], turn[-5:])
vwap_dev          = (close - vwap) / vwap            # 偏离 VWAP
```

### 3.3 时序特征算子（`factors/alphanet_features.py`）

受 AlphaNet 研究启发，设计了 6 类时序算子从原始量价序列中提取非线性组合特征，无需人工设计所有特征形式。

原始输入序列（9 个）：`open, high, low, close, volume, amount, turn, vwap, return_1d`

**6 类时序算子**

| 算子 | 含义 | 业务直觉 |
|------|------|---------|
| `ts_corr(X, Y, d)` | X、Y 过去 d 天的 Pearson 相关系数 | 量价配合度 |
| `ts_cov(X, Y, d)` | X、Y 过去 d 天的协方差 | 量价协同的绝对强度 |
| `ts_stddev(X, d)` | X 过去 d 天的标准差 | 短期波动幅度 |
| `ts_zscore(X, d)` | X 过去 d 天的 Z-score | 偏离均值的程度 |
| `ts_return(X, d)` | X 过去 d 天的涨跌幅 | 动量信号 |
| `ts_decaylinear(X, d)` | 线性衰减加权均值（近期权重更高） | 强化近期信号权重 |

**系统使用的 10 个重点组合特征**

| 特征名 | 计算方式 | 业务含义 |
|--------|---------|---------|
| `corr_close_vol_5` | ts_corr(close, volume, 5) | 5 日价量配合度 |
| `corr_close_turn_5` | ts_corr(close, turn, 5) | 5 日价格与换手配合度 |
| `corr_return_vol_5` | ts_corr(return_1d, volume, 5) | 涨幅与量的同向性 |
| `std_close_10` | ts_stddev(close, 10) | 10 日价格波动幅度 |
| `std_vol_10` | ts_stddev(volume, 10) | 10 日成交量波动 |
| `zscore_close_10` | ts_zscore(close, 10) | 价格偏离均值程度 |
| `zscore_vol_5` | ts_zscore(volume, 5) | 成交量相对水平 |
| `ret_close_10` | ts_return(close, 10) | 10 日累积收益（中期动量） |
| `decay_vol_5` | ts_decaylinear(volume, 5) | 近期放量程度 |
| `decay_close_5` | ts_decaylinear(close, 5) | 近期价格加权走势 |

### 3.4 资金流因子（`factors/capital_flow.py`）

```python
main_net_ratio  = 主力净流入额 / 总成交额          # AKShare: stock_individual_fund_flow
super_net_ratio = 超大单净流入 / 总成交额
main_net_5d     = 5 日累计主力净流入比              # 连续资金流入信号
north_net_flow  = 北向资金净流入                   # AKShare: stock_connect_flow
```

> **注意**：资金流使用前一交易日数据（T-1），不影响 T 日决策，不引入未来函数。

### 3.5 基本面过滤（用于剔除，不参与打分）

- PE_ttm < 0（亏损股剔除）
- 总市值 < 20 亿（微盘股，流动性风险）
- ST / \*ST / 退市整理 / 停牌（直接剔除）
- 近 20 日涨幅 > 40%（追高风险剔除）
- 上市不足 60 个交易日（历史数据不足，因子无法计算）

### 3.6 五因子中性化（`factors/factor_neutralize.py`）

对 XGBoost 输入因子进行截面中性化，消除行业偏差和风格暴露，提升因子 IC 稳定性。

方法：每日截面数据，以下 5 个变量为自变量做 OLS 回归，取**残差**作为中性化后的因子值：

1. 行业（申万一级行业哑变量，28 个行业）
2. 对数总市值 `ln(market_cap)`
3. 20 日动量 `mom_20d`
4. 20 日波动率 `vol_20d`
5. 5 日平均换手率 `turn_5d_avg`

---

## 第四章 量化选股层

### 4.1 全市场扫描与过滤（`models/stock_screener.py`）

每个交易日从全市场约 5000 只股票中依次过滤：

1. 剔除 ST / \*ST / 退市整理股
2. 剔除当日停牌股（成交量为 0）
3. 剔除上市不足 60 个交易日新股
4. 剔除市值低于 20 亿

过滤后通常剩余约 4200～4500 只可交易标的。

### 4.2 XGBoost 打分模型（`models/xgboost_scorer.py`）

**训练策略**
- 训练数据：2021-01 至 2023-12 历史 A 股日频数据
- 预测目标：次日截面收益率排名（分 10 档，0 最差，9 最优）
- 滚动更新：每 10 个交易日用新数据增量微调一次
- 时序交叉验证：验证集时间严格晚于训练集，最小训练窗口 240 个交易日，彻底杜绝未来函数

**关键参数**
```python
xgb_params = {
    'n_estimators': 200,
    'max_depth': 4,               # 控制过拟合
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'rank:pairwise', # RankIC 优化目标
    'eval_metric': 'ndcg',
}
```

### 4.3 候选池生成流程

1. 对全市场可交易股票向量化计算 20 个因子（单日约 3 秒）
2. 对因子值进行五因子 OLS 中性化
3. XGBoost 推理得到每只股票综合得分（0～1）
4. 降序排列取 Top30，二次过滤：剔除近 5 日重复出现的标的
5. 输出 **Top20 候选池**，附带因子摘要传入智能体层

**候选池数据结构**
```python
{
    "symbol": "600519",
    "symbol_name": "贵州茅台",
    "xgb_score": 0.847,
    "key_factors": {
        "mom_5d": 0.032,
        "corr_close_vol_5": 0.71,
        "main_net_ratio": 0.18,
        "zscore_vol_5": 1.4
    },
    "industry": "食品饮料",
    "market_cap_bn": 2100
}
```

---

## 第五章 多智能体决策层

### 5.1 设计思路

量化层提供客观的数量信号，智能体层提供语义层面的综合判断与可解释叙事。两者分工明确：

- 量化层负责「选谁」的初筛，降低 LLM 的调用成本和幻觉风险
- 智能体层负责「买不买」的深度验证，提升决策的可解释性
- 多空辩论机制引入对立视角，避免单一信号的系统性偏差

系统对 Top20 候选股使用 `asyncio` 并行分析，控制总耗时 ≤ 4 分钟。

### 5.2 各 Agent 职责

| Agent | 核心任务 | 主要数据来源 |
|-------|---------|------------|
| 市场数据分析师 | 数据预处理，构建统一数据上下文 | `stock_zh_a_hist`（日K线） |
| 技术面分析师 | 技术指标解读 + 量价特征分析 | K 线，计算 MACD/RSI/KDJ/布林带 |
| 基本面分析师 | 财务健康度评分 | `stock_financial_analysis_indicator` |
| 情绪分析师 | 新闻情绪打分（-1 到 1） | 东方财富/新浪财经新闻接口 |
| 估值分析师 | 估值安全垫判断 | PE/PB 历史分位数 |
| 宏观分析师 | 宏观环境打分（每日一次，所有标的共用） | PMI / 利率 / 货币政策数据 |
| 多方研究员 | 从多头视角提炼核心买入逻辑 | 综合前序 Agent 输出 |
| 空方研究员 | 从空头视角提炼核心风险点 | 综合前序 Agent 输出 |
| 辩论室（LLM 裁判） | 客观评估多空双方论点，输出综合置信度 | 辩论内容结构化摘要 |
| 风险管理师 | 按置信度排序，控制行业集中度 | 波动率、市值、行业分布 |
| 投资组合管理师 | 最终确认 3～5 只标的及仓位比例 | 风险管理输出 + 当日可用资金 |

### 5.3 混合置信度 C_mixed 计算

```python
# 多空研究员分别输出 0~1 的置信度
C_raw   = (bull_confidence - bear_confidence + 1) / 2  # 归一化到 [0,1]

# LLM 裁判独立打分
C_llm   = LLM 第三方评分（0~1）

# 加权融合
C_mixed = 0.6 * C_raw + 0.4 * C_llm
```

只有 `C_mixed > 0.55` 的标的才进入最终持仓候选。若全部不达标，当日返回 `[]`（空仓，不操作）。

### 5.4 LLM 选型

- **首选**：DeepSeek-V3 / DeepSeek-R1（成本低，中文金融语料充分）
- **备选**：GPT-4o 或 Claude Sonnet（通过 OpenAI Compatible 接口）
- **配置**：`.env` 中设置 `OPENAI_COMPATIBLE_BASE_URL` 和 `API_KEY`

---

## 第六章 资金管理与输出层

### 6.1 资金分配逻辑（`execution/position_sizer.py`）

**可用资金**：每日累计总资产（平台日结后更新）。初始值 500,000 元。

**仓位分配公式**
```python
total_score = sum(C_mixed_i for i in selected_stocks)
weight_i    = C_mixed_i / total_score
amount_i    = available_capital * weight_i * 0.9    # 留 10% 安全缓冲
volume_i    = int(amount_i / prev_close_i / 100) * 100  # 向下取 100 整数倍
```

**风控约束**
- 单只股票占用资金 ≤ 总资产 30%（防止集中风险）
- 最少持仓 100 股（不足则跳过，不操作）
- 当日最多操作 5 只标的

### 6.2 标准 JSON 输出（`execution/output_formatter.py`）

```json
[
  {"symbol": "600519", "symbol_name": "贵州茅台", "volume": 100},
  {"symbol": "000858", "symbol_name": "五粮液",   "volume": 200}
]
```

同时生成 `logs/YYYYMMDD_reasoning.md`，包含完整决策链路，供评审可解释性审计。

### 6.3 主入口流程（`src/main_stock_agent.py`）

> 下方为完整双层流程的概念性伪代码；当前落地版本中 Layer 3 智能体层为占位（`C_mixed = xgb_score`），量化选股 → 资金分配 → JSON 输出链路已完整实现。竞赛回测入口见 `src/backtest_stock_agent.py`。

```python
def run_daily():
    if not is_trading_day(today):
        return []

    # Layer 0~2：量化选股
    candidates = stock_screener.run(top_n=20)
    if not candidates:
        return []

    # Layer 3：多智能体决策（asyncio 并行）
    decisions = run_agents_parallel(candidates)

    # Layer 4：资金分配与输出
    result = position_sizer.allocate(decisions)
    output = output_formatter.to_json(result)
    log_decision(output, decisions)
    return output
```

---

## 第七章 可解释性与日志设计

### 7.1 每日决策日志

- `logs/YYYYMMDD_decision.json`：最终提交平台的 JSON
- `logs/YYYYMMDD_reasoning.md`：每只持仓标的的完整决策链路

**reasoning.md 示例**
```markdown
## 600519 贵州茅台  |  C_mixed: 0.704

### 量化打分
- XGB 综合得分: 0.847（当日全市场排名第 3）
- 关键驱动因子: 5日动量+3.2%, 量价相关系数0.71, 主力净流入1.2亿

### 技术面（技术面分析师）
- MACD 金叉，RSI=62（未超买），量能较5日均量放大1.3倍

### 情绪面（情绪分析师）
- 近24小时新闻情感得分: +0.6（积极）
- 主要事件: ××公告利好

### 多空辩论结论
- 多方置信度: 0.72  空方置信度: 0.28
- LLM 第三方评分: 0.68
- C_mixed = 0.6 × 0.72 + 0.4 × 0.68 = 0.704

### 资金分配
- 仓位权重: 34.5% | 买入金额: 172,500元
- 昨日收盘: 1750.0元 | 建议买入: 100 股
```

### 7.2 SHAP 因子贡献度分析（每周运行）

通过 SHAP 值量化各因子对模型决策的贡献度，直观展示「哪些因子在驱动选股结果」，满足评审「假设明确」的要求，也可直接用于答辩展示。

```python
import shap
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_today)
shap.summary_plot(shap_values, X_today, feature_names=factor_names)
```

---

## 第八章 开发路线图

### 8.1 阶段划分

| 阶段 | 优先级 | 任务 | 验收标准 |
|------|--------|------|---------|
| Phase 1 数据基础 | P0 | AKShare 封装 + SQLite 缓存 + 交易日历 | 稳定拉取近 60 日全市场 K 线，缓存命中率 > 90% |
| Phase 2 因子工程 | P0 | price_volume.py + alphanet_features.py + factor_neutralize.py | 对 500 只股票批量计算 20 个因子，耗时 < 30 秒 |
| Phase 3 选股模型 | P0 | XGBoost 训练脚本 + stock_screener.py | 输出 Top20 候选池，历史回测 IC > 3% |
| Phase 4 智能体对接 | P1 | 多 Agent 协同框架 + asyncio 并行调用 | 对 20 只候选股完成全流程，总耗时 < 4 分钟 |
| Phase 5 输出格式 | P0 | position_sizer.py + output_formatter.py | 输出合法 JSON，volume 均为 100 倍数，总金额不超可用资产 |
| Phase 6 日志审计 | P1 | reasoning.md + SHAP 分析脚本 | 每次运行生成完整决策链路文档 |
| Phase 7 自动化 | P2 | run_daily.sh + crontab / 任务计划 | 每个交易日 09:20 自动运行，09:25 前完成 |

### 8.2 关键依赖库

| 库 | 用途 | 安装 |
|----|------|------|
| akshare | 全市场行情/财务/资金流数据 | `pip install akshare` |
| xgboost | 量化打分模型 | `pip install xgboost` |
| shap | 因子贡献度分析 | `pip install shap` |
| pandas / numpy | 因子计算，向量化运算 | `pip install pandas numpy` |
| scikit-learn | OLS 中性化、特征标准化 | `pip install scikit-learn` |
| langgraph | Agent 编排框架 | `pip install langgraph` |
| lightgbm | 可选，XGBoost 替代方案 | `pip install lightgbm` |

### 8.3 工程注意事项

1. **AKShare 限频**：批量拉取时每次请求间隔 ≥ 0.3 秒，否则可能被封 IP
2. **前复权**：因子计算必须使用前复权价格（`adjust='qfq'`），否则除权日因子严重失真
3. **严禁未来函数**：T 日特征只能使用 T-1 日收盘后数据，严禁引用 T 日盘中成交数据
4. **volume 精度**：使用 `int(amount / price / 100) * 100`，避免浮点误差
5. **空仓处理**：候选池为空或全部 `C_mixed ≤ 0.55` 时返回 `[]`，系统正常空仓不报错
6. **情绪分析幻觉风险**：Sentiment Agent 必须传入原始新闻标题作为 context，禁止让 LLM 自行生成事件
7. **并发限制**：asyncio 调用 LLM 时设置 `semaphore` 最大并发数为 5，避免触发 API rate limit

---

## 第九章 策略预期与风险说明

### 9.1 量化因子历史有效性

基于学术研究与 A 股量化实践，量价类因子在 A 股市场有持续有效的 alpha：

| 策略组件 | 参考 RankIC 水平 | 参考年化超额 |
|---------|----------------|------------|
| 量价 XGBoost 合成因子 | 约 8~9% | 约 9~12% |
| 叠加多因子中性化后 | 稳定性显著提升 | 信息比率改善明显 |

> 以上为学术文献中的历史参考数据，不代表本系统未来表现。

### 9.2 主要风险

- **模型失效**：量价因子在高频风格切换行情下可能失效，需持续监控滚动 IC
- **流动性风险**：赛制按昨收结算，当日已涨停股实际无法买入，系统需过滤
- **LLM 调用失败**：网络异常时系统降级为仅依赖量化打分直接输出，保证流程不中断
- **数据延迟**：AKShare 更新时间不与交易所完全同步，建议 09:20 前完成全部数据拉取

### 9.3 迭代优化方向

- 将 XGBoost 替换为 LightGBM（速度快 3～5 倍，性能持平）
- 加入公告事件驱动模块：检测定增/回购/业绩预告触发额外信号
- 引入次日涨停预测子模型：专注高弹性标的
- 动态仓位管理：近 5 日连续亏损时自动降低总仓位上限

---

*AlphaAgent · 项目设计说明文档 · 2026-06*
