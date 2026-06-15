# AlphaAgent · A 股量化选股投资智能体

> 基于量价因子与多智能体协同的 A 股日内投资系统
> 初始资金 50 万元 · 每日输出标准 JSON 操作建议 · 日终清仓结算

**⚠️ 免责声明：本项目仅用于教育、研究与竞赛目的，不构成任何投资或实际交易建议。投资有风险，决策需谨慎。**

---

## 一、项目概述

AlphaAgent 是一个面向 A 股市场的**量化选股 + 智能体决策**双层投资系统。每个交易日从沪深两市全市场（约 5000 只）股票中自动扫描、过滤、打分，筛选出候选标的，按置信度加权分配资金，最终输出符合赛制要求的 JSON 操作建议。

系统采用双层架构：

- **量化层（已实现）**：量价因子 + AlphaNet 时序特征 → 五因子中性化 → XGBoost 打分 → Top-N 候选池。客观、可回测、可复现。
- **智能体层（设计中 / 占位）**：对候选标的调用多个专业 Agent（技术面、基本面、情绪、估值、宏观）协同分析，经多空辩论输出混合置信度 `C_mixed`。当前为占位实现，`C_mixed` 直接取量化得分。

> 因子构建与选股策略的详细可解释性说明见 [因子与选股策略说明.md](因子与选股策略说明.md)。
> 完整系统设计文档见 [PROJECT_SPEC.md](PROJECT_SPEC.md)。

---

## 二、系统流水线

```
每日触发
   ↓
Layer 0  数据采集   baostock（主） / AKShare（备）拉取全市场前复权 K 线
   ↓
Layer 1  因子计算   量价因子(13) + AlphaNet 时序特征(10)
   ↓
Layer 2  量化选股   五因子 OLS 中性化 → XGBoost 打分 → Top-N 候选池
   ↓
Layer 3  智能体决策 多 Agent 分析 → 混合置信度 C_mixed（当前为占位实现）
   ↓
Layer 4  输出执行   按 C_mixed 加权分配资金 → volume 取 100 整数倍 → JSON + 审计日志
```

---

## 三、数据源说明

| 数据源 | 用途 | 状态 |
|--------|------|------|
| **baostock** | 全市场股票列表、日频前复权 K 线 | **主数据源** |
| AKShare | 实时行情、财务、新闻；K 线备用 | 备用 / 降级 |

> **为什么以 baostock 为主**：AKShare 底层依赖东方财富 HTTPS 接口，在部分网络环境（如本地开启系统代理）下易出现 `ProxyError` / `RemoteDisconnected`。baostock 使用独立协议，更稳定。代码会优先走 baostock，失败时自动回退到 AKShare（见 [stock_screener.py](src/models/stock_screener.py) 与 [price_volume.py](src/factors/price_volume.py)）。

---

## 四、安装与环境

### 1. 创建虚拟环境并安装依赖

项目使用 `pyproject.toml` 管理依赖，推荐 `uv`：

```bash
uv sync
```

或使用已有的 `venv`（Windows）：

```powershell
# 依赖已安装在 venv 中；如需重装，先导出再安装：
uv export --format requirements-txt > requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

核心依赖：`pandas`、`numpy`、`baostock`、`akshare`、`xgboost`、`scikit-learn`、`matplotlib`。

### 2. 配置环境变量（智能体层可选）

复制 `.env.example` 为 `.env`，量化选股层无需任何 API Key 即可运行。仅当启用智能体层时需配置 LLM：

```env
# LLM（智能体层，可选）
OPENAI_COMPATIBLE_API_KEY=your_key
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1
OPENAI_COMPATIBLE_MODEL=deepseek-chat

# 代理轮询（AKShare 用，默认 direct 即不走代理）
AKSHARE_PROXY_LIST=direct
```

---

## 五、使用指南

> Windows 下请用 `.\venv\Scripts\python.exe` 替换下方的 `python`。

### 1. 每日选股（生成当日操作建议）

```bash
python -m src.main_stock_agent --date 20260613 --top-n 20 --capital 500000 --skip-llm
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--date` | 交易日期 YYYYMMDD | 今天 |
| `--top-n` | 候选池大小 | 20 |
| `--capital` | 可用资金 | 500000 |
| `--skip-llm` | 跳过智能体层（仅量化选股） | false |

### 2. 竞赛口径回测

按赛制结算（前一交易日收盘价买入、当日收盘价卖出、日终清仓、次日资金=累计总资产）。

```bash
# MVP 快速验证：随机 500 只 × 最近 3 个交易日
python -m src.backtest_stock_agent --mvp

# 完整回测
python -m src.backtest_stock_agent --start 20260518 --end 20260613 --capital 500000 --top-n 20
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--mvp` | 快速模式（500 只 × 3 日） | - |
| `--start` / `--end` | 回测区间 YYYYMMDD | 20260518 / 今天 |
| `--capital` | 初始资金 | 500000 |
| `--top-n` | 每日候选池大小 | 20 |
| `--max-stocks` | 限制股票池大小（加速测试） | 全市场 |
| `--cache-dir` | K 线缓存目录（断点续传） | data/prefetch_cache |
| `--model-path` | XGBoost 模型路径（不存在时等权降级） | data/xgb_scorer.model |
| `--offline` | 离线模式：只用本地缓存股票、完全不联网 | false |

**断点续传**：每只股票拉完即写盘缓存，中断后重新运行同一命令自动跳过已缓存股票；网络失败自动重试（指数退避）。

### 3. 训练 XGBoost 模型

用历史数据训练排序打分模型（特征与选股/回测端完全一致：量价 + AlphaNet + 中性化；严格防未来函数）。

```bash
# 先小样本验证流程（约 10-15 分钟）
python -m src.models.train_xgboost --start 20240101 --end 20241231 --max-stocks 300

# 完整训练，保存到 data/xgb_scorer.model
python -m src.models.train_xgboost --start 20220101 --end 20251231 --output data/xgb_scorer.model
```

训练完成后，`main_stock_agent` 与 `backtest_stock_agent` 会自动加载 `data/xgb_scorer.model`（可用 `--model-path` 指定）；模型不存在时自动降级为等权打分。

### 4. 回测输出

回测结束后在 `logs/` 下生成：

| 文件 | 内容 |
|------|------|
| `backtest_<tag>.csv` | 每日资产 / 盈亏 / 收益率明细 |
| `backtest_<tag>.png` | 总资产走势 + 日收益 / 累计收益曲线 |
| `advice_<tag>/YYYYMMDD.json` | **每个交易日**的操作建议 |
| `advice_<tag>.json` | 所有交易日操作建议汇总 |

**操作建议 JSON 格式（赛制标准）**：

```json
[
  {"symbol": "600519", "symbol_name": "贵州茅台", "volume": 100},
  {"symbol": "000858", "symbol_name": "五粮液",   "volume": 200}
]
```

---

## 六、项目结构

```
invest_aiagent/
├── src/
│   ├── factors/                      # 量化因子体系
│   │   ├── price_volume.py           # 量价因子（动量/波动/换手/量价相关）+ K 线拉取
│   │   ├── alphanet_features.py      # 6 类时序算子 → 10 个 AlphaNet 特征
│   │   ├── factor_neutralize.py      # 五因子 OLS 截面中性化
│   │   └── capital_flow.py           # 资金流因子（已定义，尚未接入选股流程）
│   ├── models/                       # 量化模型
│   │   ├── stock_screener.py         # 全市场扫描 → 过滤 → 因子 → 中性化 → 打分 → 候选池
│   │   ├── xgboost_scorer.py         # XGBoost 排序打分（训练/推理/微调）
│   │   └── train_xgboost.py          # 模型训练脚本（构建训练集 + 时序切分 + 训练）
│   ├── execution/                    # 输出执行层
│   │   ├── position_sizer.py         # 按 C_mixed 加权分配资金，volume 取整
│   │   └── output_formatter.py       # 标准 JSON 输出 + reasoning.md 审计日志
│   ├── tools/                        # 数据接入与工具
│   │   ├── baostock_client.py        # baostock 封装（登录复用/重试）
│   │   ├── akshare_cache.py          # AKShare + SQLite 缓存层
│   │   └── ...
│   ├── network/
│   │   └── proxy_manager.py          # 代理轮询 + AKShare no-proxy 补丁
│   ├── agents/                       # 多智能体模块（Layer 3，设计中/占位）
│   ├── main_stock_agent.py           # ★ 竞赛主入口（每日选股）
│   ├── backtest_stock_agent.py       # ★ 竞赛口径回测
│   ├── main.py                       # 原项目入口（单股票 LLM 分析，保留）
│   └── backtester.py                 # 原项目回测（单股票，保留）
├── PROJECT_SPEC.md                   # 系统完整设计文档
├── 因子与选股策略说明.md              # 因子构建与选股策略可解释性说明
├── pyproject.toml
└── README.md
```

> `src/main.py` 与 `src/backtester.py` 来自原始开源项目（单股票多智能体 LLM 对冲基金），本项目保留它们作为智能体层的参考实现，竞赛主流程不依赖它们。

---

## 七、当前实现状态

| 模块 | 状态 | 说明 |
|------|------|------|
| 全市场扫描 + ST/市值/PE 过滤 | ✅ 已实现 | baostock 主，约 5000 → 4970 只可交易 |
| 量价因子（13 个） | ✅ 已实现 | [price_volume.py](src/factors/price_volume.py) |
| AlphaNet 时序特征（10 个） | ✅ 已实现 | [alphanet_features.py](src/factors/alphanet_features.py) |
| 五因子 OLS 中性化 | ✅ 已实现 | [factor_neutralize.py](src/factors/factor_neutralize.py) |
| XGBoost 打分 + 训练脚本 | ✅ 已实现 | [train_xgboost.py](src/models/train_xgboost.py)；未训练模型时自动等权 `0.5` 降级 |
| 模型自动加载 | ✅ 已实现 | 主流程与回测自动加载 `data/xgb_scorer.model`（`--model-path` 可指定） |
| 资金分配 + JSON 输出 | ✅ 已实现 | [position_sizer.py](src/execution/position_sizer.py) |
| 竞赛口径回测 + 每日建议 | ✅ 已实现 | [backtest_stock_agent.py](src/backtest_stock_agent.py) |
| 资金流因子 | ⚠️ 已定义，未接入 | [capital_flow.py](src/factors/capital_flow.py) |
| 多智能体决策层 | ⚠️ 占位实现 | `C_mixed = xgb_score` |

> **重要**：XGBoost 尚未训练模型时，全市场股票得分均为 `0.5`（等权），选股相当于在中性化因子排序上的基线。要体现量化 alpha，需先训练模型并通过 `StockScreener.set_model()` 加载。

---

## 八、致谢与许可

本项目的智能体层参考并修改自 [A_Share_investment_Agent](https://github.com/24mlight/A_Share_investment_Agent)（其本身改编自 [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)）。量化选股、因子体系、竞赛回测部分为本项目新增。

许可证详见 [LICENSE](LICENSE)（原始代码 MIT，新增代码 GPL v3 + 非商业条款）。
