# AlphaAgent · 面向 A 股的量化选股智能体

AlphaAgent 是一个面向 A 股市场的量化选股 Agent 项目。系统以日频行情数据为基础，完成股票池构建、因子计算、因子有效性检验、选股排序、资金分配、回测验证和操作建议输出，适用于量化投资研究、竞赛展示和 Agent 产品设计说明。

> 免责声明：本项目仅用于教育、研究和竞赛展示，不构成任何投资建议。投资有风险，实盘交易需自行承担风险。

## 当前最新结论

经过多轮因子有效性验证和 2024、2025、2026 多窗口回测，项目最终默认策略为：

```text
score_method = train_ic_blend
allocation_method = score
position_count = 3
max_drawdown_stop = 0
```

含义：

- `train_ic_blend`：使用训练期 IC 验证过的因子方向进行排序。
- `score`：按股票得分进行资金加权分配。
- `position_count = 3`：每日最多持有 3 只股票。
- `max_drawdown_stop = 0`：默认不启用硬停手机制；风控参数保留为可选实验项。

最终验证显示，旧的 `adaptive_blend` 默认策略在扩展 2026 样本中表现不稳定，已不再作为默认策略。当前 README 只保留最新可交付版本的说明。

## 最新回测表现

回测口径：初始资金 500,000 元，日频选股，前一交易日收盘后生成信号，当日收盘价结算，日终清仓，使用本地缓存离线复现。

| 窗口 | 股票数 | 交易日 | 累计收益率 | 年化收益率 | 最大回撤 | 夏普比率 | 索泰诺比率 | 卡玛比率 | 胜率 | 盈亏比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 全年 | 500 | 241 | 17.77% | 18.65% | 21.93% | 0.83 | 1.35 | 0.85 | 51.04% | 1.12 |
| 2025 全年 | 500 | 242 | 19.75% | 20.64% | 7.11% | 1.44 | 1.91 | 2.90 | 50.41% | 1.28 |
| 2026 上半年 | 500 | 115 | 11.34% | 26.54% | 7.63% | 1.28 | 2.08 | 3.48 | 51.30% | 1.20 |

指标文件：

- `logs/factor_analysis/final_performance_metrics.csv`
- `logs/factor_analysis/final_default_comparison.csv`
- `logs/factor_analysis/validation_report.md`

### 产品优势

- 2025 全年卡玛比率为 2.90，说明收益与回撤之间的平衡较好。
- 2026 上半年最大回撤为 7.63%，卡玛比率为 3.48，说明扩展样本中仍保持较好的风险收益比。
- 胜率约 50%，但盈亏比大于 1，说明策略优势不是依赖高胜率，而是依靠盈利日略大于亏损日的结构获得正收益。
- 2024、2025、2026 三个窗口均为正收益，证明当前默认策略具备跨窗口稳定性。

需要注意：2024 全年最大回撤为 21.93%，说明策略在市场风格剧烈切换时仍有明显回撤风险。作品展示时建议诚实说明这一点，并作为后续风控优化方向。

## 项目结构

```text
invest_aiagent/
├── src/
│   ├── backtest_stock_agent.py      # 竞赛口径回测主入口，当前默认策略在这里
│   ├── main_stock_agent.py          # 每日选股入口
│   ├── agents/                      # 多智能体分析模块
│   ├── execution/                   # 仓位分配和输出格式化
│   ├── factors/                     # 量价因子、AlphaNet 特征、中性化
│   ├── models/                      # 股票筛选器、XGBoost 训练与推理
│   ├── tools/                       # baostock、AKShare、市场数据工具
│   └── utils/                       # 配置、日志、序列化等工具
├── scripts/
│   ├── analyze_factor_ic.py         # 因子 IC 分析脚本
│   ├── agent_status_check.py        # Agent 状态检查
│   └── run_pipeline_check.py        # 流程检查脚本
├── data/
│   ├── train_cache/                 # 2023-08 至 2025-12 历史 K 线缓存
│   ├── prefetch_cache/              # 近期行情缓存
│   ├── factor_cache/                # 近期因子缓存
│   └── factor_cache_train/          # 训练期因子缓存
├── logs/
│   ├── factor_analysis/             # 最终指标、因子 IC、验证报告
│   ├── full_2024_default_final_s500/
│   ├── full_2025_default_final_s500/
│   └── h1_2026_default_final_s500/
├── 作品设计书_撰写稿.md              # 最新作品设计书草稿
├── pyproject.toml
└── README.md
```

## 环境安装

推荐使用项目已有的 `venv`，或通过 `uv` / `poetry` 安装依赖。

### 使用已有虚拟环境

Windows PowerShell：

```powershell
.\venv\Scripts\python.exe -m pip install -e .
```

### 使用 uv

```powershell
uv sync
```

核心依赖包括：

- `pandas`
- `numpy`
- `baostock`
- `akshare`
- `xgboost`
- `scikit-learn`
- `matplotlib`
- `fastapi`
- `langchain`

## 快速运行

### 1. 使用最终默认策略跑 2026 上半年

```powershell
.\venv\Scripts\python.exe -m src.backtest_stock_agent `
  --start 20260101 `
  --end 20260630 `
  --offline `
  --interval 0 `
  --cache-dir data\prefetch_cache `
  --factor-cache-dir data\factor_cache `
  --no-plot `
  --quiet-advice `
  --max-stocks 500 `
  --log-dir logs\h1_2026_default_final_s500
```

这条命令不显式传策略参数，因为当前默认值已经是最终验证版本：

```text
--score-method train_ic_blend
--allocation-method score
--position-count 3
--max-drawdown-stop 0
```

### 2. 跑 2024 全年离线回测

```powershell
.\venv\Scripts\python.exe -m src.backtest_stock_agent `
  --start 20240101 `
  --end 20241231 `
  --offline `
  --interval 0 `
  --cache-dir data\train_cache `
  --factor-cache-dir data\factor_cache_train `
  --no-plot `
  --quiet-advice `
  --max-stocks 500 `
  --log-dir logs\full_2024_default_final_s500
```

### 3. 跑 2025 全年离线回测

```powershell
.\venv\Scripts\python.exe -m src.backtest_stock_agent `
  --start 20250101 `
  --end 20251231 `
  --offline `
  --interval 0 `
  --cache-dir data\train_cache `
  --factor-cache-dir data\factor_cache_train `
  --no-plot `
  --quiet-advice `
  --max-stocks 500 `
  --log-dir logs\full_2025_default_final_s500
```

## 常用参数

| 参数 | 说明 | 当前默认 |
|---|---|---|
| `--score-method` | 排序方法，可选 `xgb`、`ic_blend`、`recent_ic_blend`、`train_ic_blend`、`ensemble_blend`、`adaptive_blend` | `train_ic_blend` |
| `--allocation-method` | 资金分配方式，可选 `equal` 或 `score` | `score` |
| `--position-count` | 每日持仓数量 | `3` |
| `--max-drawdown-stop` | 回撤停手阈值，0 表示关闭 | `0` |
| `--max-stocks` | 回测股票池数量限制 | 无限制 |
| `--offline` | 只使用本地缓存，不联网 | 关闭 |
| `--cache-dir` | K 线缓存目录 | `data/prefetch_cache` |
| `--factor-cache-dir` | 因子缓存目录 | 无 |
| `--no-plot` | 跳过图表生成，加快批量回测 | 关闭 |
| `--quiet-advice` | 不在终端打印每日 JSON 建议 | 关闭 |

## 因子与策略逻辑

### 因子体系

系统主要使用量价类因子：

- 动量因子：短期和中期收益率。
- 反转因子：短期反转收益。
- 波动因子：价格波动、成交量波动。
- 成交量因子：成交量均值、量比、成交量衰减。
- 换手因子：换手率及换手均值。
- 价量关系因子：价格与成交量、收益与成交量相关性。

### 最终排序逻辑

当前默认策略 `train_ic_blend` 使用训练期 IC 验证后的因子方向构造综合排序分数。相比单纯依赖 XGBoost 分数，该方法更容易解释，并且在 2024、2025、2026 多个窗口中表现更稳定。

### 资金分配逻辑

当前默认使用 `score` 加权：

1. 每日选择综合得分最高的 3 只股票。
2. 按得分权重分配资金。
3. 单只股票买入数量向下取 100 股整数倍。
4. 日终按收盘价结算并清仓。

## 可解释性方案

项目的可解释性分为三层：

1. 因子级解释：每个候选股票得分来自明确的量价因子。
2. 策略级解释：通过 IC、收益率、回撤、夏普、索泰诺、卡玛、胜率、盈亏比解释策略优缺点。
3. 迭代级解释：保留策略从 XGBoost、自适应、IC blend 到 Train-IC 默认策略的验证过程，证明策略是由数据反馈驱动调整。

## Agent 架构

```text
数据采集 Agent
  -> 因子计算 Agent
  -> 策略评估 Agent
  -> 选股决策 Agent
  -> 资金分配与执行 Agent
  -> 解释与报告 Agent
```

模块职责：

- 数据采集 Agent：维护 K 线缓存，支持联网拉取和离线复现。
- 因子计算 Agent：生成量价因子和截面因子缓存。
- 策略评估 Agent：计算 IC 和回测评价指标。
- 选股决策 Agent：根据策略得分输出 Top 3 标的。
- 资金分配与执行 Agent：生成买入数量和 JSON 操作建议。
- 解释与报告 Agent：输出回测报告、指标表和作品设计书素材。

## 最新产物

| 文件 | 说明 |
|---|---|
| `作品设计书_撰写稿.md` | 当前最新作品设计书草稿 |
| `logs/factor_analysis/final_performance_metrics.csv` | 夏普、索泰诺、卡玛、胜率、盈亏比等最终指标 |
| `logs/factor_analysis/final_default_comparison.csv` | 最终默认策略跨窗口结果 |
| `logs/factor_analysis/validation_report.md` | 策略验证报告 |
| `logs/full_2024_default_final_s500/` | 2024 全年最终回测结果 |
| `logs/full_2025_default_final_s500/` | 2025 全年最终回测结果 |
| `logs/h1_2026_default_final_s500/` | 2026 上半年最终回测结果 |

## 旧内容处理说明

本 README 已更新为项目当前唯一推荐入口。早期实验内容仍可能保留在 `logs/factor_analysis/` 中，用于追溯策略迭代过程，但不再作为默认策略依据。

当前最终版本只建议引用以下结论：

- 默认策略：`train_ic_blend + score allocation + 3 positions`
- 最终回测：2024 全年、2025 全年、2026 上半年
- 最终指标文件：`final_performance_metrics.csv`
- 最终设计书：`作品设计书_撰写稿.md`

## 风险与后续优化

当前系统仍有以下限制：

- 2024 全年最大回撤较高，说明市场风格切换时仍存在明显风险。
- 当前回测未完整模拟交易成本、滑点、涨跌停无法成交等实盘约束。
- 2026 样本目前使用本地缓存可靠覆盖区间，后续可继续补足更长 2026 数据。
- LLM 多智能体层仍可继续增强基本面、新闻情绪和宏观分析。

后续优化方向：

- 增加交易成本和滑点模拟。
- 增加市场状态识别模块。
- 增加基本面和新闻情绪 Agent。
- 做成交约束、涨跌停过滤和风险预算控制。
- 将回测和每日选股报告部署到云端定时运行。

## 作品展示建议

海报和演示视频建议突出：

- 项目不是单一模型，而是完整的量化选股 Agent 闭环。
- 系统具备数据采集、因子计算、策略验证、资金分配、报告输出能力。
- 项目经过负收益策略淘汰和默认策略重选，体现自动诊断和迭代能力。
- 最终策略在 2024、2025 和 2026 上半年样本中均为正收益。
- 产品优势是可解释、可复现和风险收益比较稳健。

一句话总结：

> AlphaAgent 将多因子量化选股和 Agent 架构结合，通过可解释因子、离线复现、多指标回测和策略迭代，形成了一个可展示、可验证、可继续扩展的 A 股智能选股系统。
