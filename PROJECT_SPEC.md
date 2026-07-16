# AlphaAgent 项目结构与系统设计说明

本文档是当前项目的最新系统设计说明。旧版中以 XGBoost 作为默认打分、以 adaptive_blend 作为默认策略、或以 5 只股票作为默认持仓的内容均已废弃。当前实际入口和最终结论以本文档与 `README.md` 为准。

## 1. 产品定位

AlphaAgent 是一个面向 A 股市场的量化选股智能体。系统通过数据采集、因子计算、因子有效性验证、选股排序、资金分配、回测评估和解释报告，形成一个可复现、可审计、可展示的投资研究闭环。

项目重点不是单次预测，而是“可验证的策略迭代”：当某一策略在更长窗口中累计收益为负或回撤不稳定时，系统通过回测指标和因子 IC 诊断问题，并替换为更稳健的默认配置。

## 2. 当前最终默认策略

```text
score_method = train_ic_blend
allocation_method = score
position_count = 3
max_drawdown_stop = 0
```

说明：

- `train_ic_blend`：使用训练期 IC 验证后的因子方向进行综合排序。
- `score`：按股票综合得分进行资金加权，而不是简单等权。
- `position_count = 3`：每日最多持有 3 只股票。多窗口测试显示，3 只持仓在收益和回撤之间更均衡。
- `max_drawdown_stop = 0`：默认不启用简单停手机制。测试发现固定冷却式停手容易在下跌后错过修复，因此仅保留为可选参数。

## 3. 最新回测结论

| 窗口 | 股票数 | 交易日 | 累计收益率 | 最大回撤 | 说明 |
|---|---:|---:|---:|---:|---|
| 2024 全年 | 500 | 241 | +17.77% | 21.93% | 收益为正，但回撤较高 |
| 2025 全年 | 500 | 242 | +19.75% | 7.11% | 风险收益比较好 |
| 2026 上半年 | 500 | 115 | +11.34% | 7.63% | 扩展至完整上半年验证窗口 |

扩展指标见：

- `logs/factor_analysis/final_performance_metrics.csv`
- `logs/factor_analysis/final_default_comparison.csv`
- `logs/factor_analysis/validation_report.md`

## 4. 总体架构

```text
数据采集层
  -> 因子计算层
  -> 因子评估与策略诊断层
  -> 选股决策层
  -> 资金分配与执行层
  -> 解释与报告层
```

### 4.1 数据采集层

职责：

- 使用 baostock / AKShare 获取 A 股日频行情。
- 将 K 线数据写入本地缓存。
- 支持断点续传和离线回测。
- 处理非交易日起止日期，避免节假日导致缓存误判。

相关文件：

- `src/tools/baostock_client.py`
- `src/models/stock_screener.py`
- `src/backtest_stock_agent.py`

### 4.2 因子计算层

职责：

- 基于前复权日频 K 线计算量价因子。
- 构造 AlphaNet 风格的时序特征。
- 对截面因子进行缓存，加快重复回测。

相关文件：

- `src/factors/price_volume.py`
- `src/factors/alphanet_features.py`
- `src/factors/factor_neutralize.py`
- `src/backtest_stock_agent.py`

### 4.3 因子评估与策略诊断层

职责：

- 计算因子 IC、收益、回撤、夏普、索泰诺、卡玛、胜率、盈亏比。
- 对比不同排序方法和资金分配方法。
- 当累计收益为负时，定位问题并调整默认策略。

相关文件：

- `scripts/analyze_factor_ic.py`
- `logs/factor_analysis/final_performance_metrics.csv`
- `logs/factor_analysis/final_default_comparison.csv`

### 4.4 选股决策层

职责：

- 根据 `train_ic_blend` 生成股票综合得分。
- 每日选择得分最高的 3 只股票。
- 保留 `xgb`、`ic_blend`、`recent_ic_blend`、`ensemble_blend`、`adaptive_blend` 等方法作为实验选项。

相关文件：

- `src/backtest_stock_agent.py`
- `src/models/stock_screener.py`

### 4.5 资金分配与执行层

职责：

- 根据得分权重分配资金。
- 限制单只股票最大资金占比。
- 将买入数量向下取整到 100 股。
- 输出每日 JSON 操作建议。

相关文件：

- `src/execution/position_sizer.py`
- `src/execution/output_formatter.py`

### 4.6 解释与报告层

职责：

- 保存回测明细、每日建议和指标汇总。
- 生成作品设计书素材。
- 解释策略迭代逻辑和风险收益特征。

相关文件：

- `README.md`
- `作品设计书_撰写稿.md`
- `logs/factor_analysis/validation_report.md`

## 5. 关键策略迭代记录

### 5.1 被淘汰的旧默认策略

旧版本曾测试过：

- `xgb`
- `ic_blend`
- `ensemble_blend`
- `adaptive_blend`
- 回撤停手机制
- 高价股过滤
- 1/3/5/10/20 只持仓

其中 `adaptive_blend + equal allocation + 5 positions + 8% risk stop` 在 2026 扩展窗口中收益为 -6.20%，因此不再作为默认策略。

### 5.2 当前默认策略的选择原因

`train_ic_blend + score allocation + 3 positions` 在以下窗口中均为正收益：

- 2024 全年：+17.77%
- 2025 全年：+19.75%
- 2026 上半年：+11.34%

与更激进的 IC blend 相比，当前默认策略收益更平滑、回撤更可控；与 XGBoost 或 adaptive_blend 相比，可解释性更强、跨窗口稳定性更好。

## 6. 当前项目目录

```text
src/
├── backtest_stock_agent.py      # 回测主入口与默认策略
├── main_stock_agent.py          # 每日选股入口
├── agents/                      # 多智能体模块
├── execution/                   # 仓位分配与输出
├── factors/                     # 因子计算
├── models/                      # 股票筛选器与模型
├── tools/                       # 数据源工具
└── utils/                       # 配置、日志、序列化

scripts/
├── analyze_factor_ic.py         # 因子 IC 分析
├── agent_status_check.py
└── run_pipeline_check.py

logs/factor_analysis/
├── final_performance_metrics.csv
├── final_default_comparison.csv
└── validation_report.md
```

## 7. 风险与待优化点

当前仍需注意：

- 2024 全年最大回撤较高，说明市场风格切换时仍有风险。
- 当前回测未完整模拟手续费、滑点、涨跌停无法成交等实盘约束。
- 2026 当前使用本地缓存可靠覆盖窗口，未来可继续补齐更长 2026 数据。
- LLM 多智能体层已有结构，但尚可进一步接入基本面、新闻情绪和宏观分析。

后续优化方向：

- 增加交易成本和滑点模型。
- 增加涨跌停和停牌过滤。
- 增加市场状态识别 Agent。
- 增加基本面和新闻情绪 Agent。
- 将回测与每日选股报告部署到云端定时运行。
