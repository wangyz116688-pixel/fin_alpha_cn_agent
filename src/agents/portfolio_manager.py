from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from typing import Any, Dict
from src.utils.logging_config import setup_logger

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.prompt_loader import load_prompt, format_prompt

# 初始化 logger
logger = setup_logger('portfolio_management_agent')

##### Portfolio Management Agent #####

# Helper function to get the latest message by agent name


def get_latest_message_by_name(messages: list, name: str, *, log_missing: bool = True):
    for msg in reversed(messages):
        if msg.name == name:
            return msg
    if log_missing:
        logger.warning(
            f"Message from agent '{name}' not found in portfolio_management_agent.")
    # Return a dummy message object or raise an error, depending on desired handling
    # For now, returning a dummy message to avoid crashing, but content will be None.
    return HumanMessage(content=json.dumps({"signal": "error", "details": f"Message from {name} not found"}), name=name)


def _normalize_macro_news_payload(raw_value: Any) -> Dict[str, Any]:
    payload = {
        "index": "沪深300指数",
        "signal": "neutral",
        "confidence": 0.5,
        "score": 50,
        "summary": "宏观新闻分析不可用。",
        "key_drivers": ["暂无"],
        "key_risks": ["暂无"],
        "actionable_insight": "暂无",
        "news_count": 0,
        "from_cache": False,
    }
    if isinstance(raw_value, dict):
        payload.update({
            "index": raw_value.get("index", payload["index"]),
            "signal": raw_value.get("signal", payload["signal"]),
            "confidence": raw_value.get("confidence", payload["confidence"]),
            "score": raw_value.get("score", payload["score"]),
            "summary": raw_value.get("summary", payload["summary"]),
            "key_drivers": raw_value.get("key_drivers", payload["key_drivers"]) or ["暂无"],
            "key_risks": raw_value.get("key_risks", payload["key_risks"]) or ["暂无"],
            "actionable_insight": raw_value.get("actionable_insight", payload["actionable_insight"]),
            "news_count": raw_value.get("news_count", payload["news_count"]),
            "from_cache": raw_value.get("from_cache", payload["from_cache"]),
        })
    elif isinstance(raw_value, str) and raw_value.strip():
        payload["summary"] = raw_value.strip()
    return payload


def _parse_decision_json(raw_text: str) -> Dict[str, Any]:
    if raw_text is None:
        raise ValueError("LLM returned None for portfolio decision")
    text = raw_text.strip()
    if not text:
        raise ValueError("LLM returned empty string for portfolio decision")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Remove surrounding code fences like ```json ... ```
        if text.startswith("```"):
            fence_match = re.match(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL | re.IGNORECASE)
            if fence_match:
                candidate = fence_match.group(1).strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    text = candidate  # continue with fallback extraction
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
    raise ValueError("Unable to parse LLM decision JSON")


@agent_endpoint("portfolio_management", "负责投资组合管理和最终交易决策")
def portfolio_management_agent(state: AgentState):
    """Responsible for portfolio management"""
    agent_name = "portfolio_management_agent"
    logger.info("📊 Portfolio Manager start (messages=%d)", len(state["messages"]))

    # Log raw incoming messages
    # logger.info(
    # f"--- DEBUG: {agent_name} RAW INCOMING messages: {[msg.name for msg in state['messages']]} ---")
    # for i, msg in enumerate(state['messages']):
    #     logger.info(
    #         f"  DEBUG RAW MSG {i}: name='{msg.name}', content_preview='{str(msg.content)[:100]}...'")

    # Clean and unique messages by agent name, taking the latest if duplicates exist
    # This is crucial because this agent is a sink for multiple paths.
    unique_incoming_messages = {}
    for msg in state["messages"]:
        # Keep overriding with later messages to get the latest by name
        unique_incoming_messages[msg.name] = msg

    cleaned_messages_for_processing = list(unique_incoming_messages.values())
    logger.info(
        "📥 Aggregated最新消息: %s",
        ", ".join(sorted(unique_incoming_messages.keys())),
    )
    # logger.info(
    # f"--- DEBUG: {agent_name} CLEANED messages for processing: {[msg.name for msg in cleaned_messages_for_processing]} ---")

    show_workflow_status(f"{agent_name}: --- Executing Portfolio Manager ---")
    show_reasoning_flag = state["metadata"]["show_reasoning"]
    data = state["data"]
    portfolio = data["portfolio"]

    # Get messages from other agents using the cleaned list
    technical_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "technical_analyst_agent")
    fundamentals_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "fundamentals_agent")
    sentiment_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "sentiment_agent")
    valuation_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "valuation_agent")
    risk_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "risk_management_agent", log_missing=False)
    tool_based_macro_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_analyst_agent")  # This is the main analysis path output

    # Extract content, handling potential None if message not found by get_latest_message_by_name
    technical_content = technical_message.content if technical_message else json.dumps(
        {"signal": "error", "details": "Technical message missing"})
    fundamentals_content = fundamentals_message.content if fundamentals_message else json.dumps(
        {"signal": "error", "details": "Fundamentals message missing"})
    sentiment_content = sentiment_message.content if sentiment_message else json.dumps(
        {"signal": "error", "details": "Sentiment message missing"})
    valuation_content = valuation_message.content if valuation_message else json.dumps(
        {"signal": "error", "details": "Valuation message missing"})
    risk_content = risk_message.content if risk_message else json.dumps(
        {"signal": "error", "details": "Risk message missing"})
    tool_based_macro_content = tool_based_macro_message.content if tool_based_macro_message else json.dumps(
        {"signal": "error", "details": "Tool-based Macro message missing"})

    # Ensure risk management context is available even if the message wasn't present in the graph state.
    if data.get("risk_analysis"):
        try:
            parsed_risk = json.loads(risk_content)
        except Exception:
            parsed_risk = None
        if not parsed_risk or parsed_risk.get("details") == "Risk message missing":
            risk_content = json.dumps(data["risk_analysis"])

    macro_news_payload = _normalize_macro_news_payload(
        data.get("macro_news_analysis_result"))
    macro_news_summary_line = (
        f"指数: {macro_news_payload['index']} | "
        f"信号: {macro_news_payload['signal']} | "
        f"置信度: {macro_news_payload['confidence']:.2f} | "
        f"分数: {macro_news_payload['score']}"
    )
    macro_news_prompt_block = (
        f"{macro_news_summary_line}\n"
        f"摘要: {macro_news_payload['summary']}\n"
        f"关键驱动: {', '.join(macro_news_payload['key_drivers'])}\n"
        f"关键风险: {', '.join(macro_news_payload['key_risks'])}\n"
        f"操作提示: {macro_news_payload['actionable_insight']}\n"
        f"新闻条数: {macro_news_payload.get('news_count', 0)}, "
        f"来源: {'缓存' if macro_news_payload.get('from_cache') else '实时'}"
    )
    logger.info(
        "🌍 宏观摘要: signal=%s score=%s cache=%s news=%d",
        macro_news_payload["signal"],
        macro_news_payload["score"],
        macro_news_payload.get("from_cache"),
        macro_news_payload.get("news_count", 0),
    )
    market_wide_news_summary_content = macro_news_prompt_block
    # Optional: also try to get the message object for consistency in agent_signals, though data field is primary source
    macro_news_agent_message_obj = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_news_agent")

    system_message_content = load_prompt("prompts/portfolio_manager/system.md")
    system_message = {
        "role": "system",
        "content": system_message_content
    }

    user_message_content = format_prompt(
        "prompts/portfolio_manager/user.md",
        technical_content=technical_content,
        fundamentals_content=fundamentals_content,
        sentiment_content=sentiment_content,
        valuation_content=valuation_content,
        risk_content=risk_content,
        tool_based_macro_content=tool_based_macro_content,
        market_wide_news_summary_content=market_wide_news_summary_content,
        portfolio_cash=f"{portfolio['cash']:.2f}",
        portfolio_stock=portfolio["stock"],
    )
    user_message = {
        "role": "user",
        "content": user_message_content
    }

    show_agent_reasoning(
        agent_name, f"Preparing LLM. User msg includes: TA, FA, Sent, Val, Risk, GeneralMacro, MarketNews.")

    llm_interaction_messages = [system_message, user_message]
    logger.info("🤖 Portfolio Manager 调用 LLM 生成决策...")
    llm_response_content = log_llm_interaction(state)(get_chat_completion)(
        llm_interaction_messages
    )
    logger.info("✅ Portfolio Manager LLM 调用完成 (响应长度=%s)", len(llm_response_content or ""))

    if llm_response_content is None:
        show_agent_reasoning(
            agent_name, "LLM call failed. Using default conservative decision.")
        logger.error("❌ Portfolio Manager LLM 调用失败，使用默认 hold 决策")
        # Ensure the dummy response matches the expected structure for agent_signals
        llm_response_content = json.dumps({
            "action": "hold",
            "quantity": 0,
            "confidence": 0.7,
            "agent_signals": [
                {"agent_name": "technical_analysis",
                    "signal": "neutral", "confidence": 0.0},
                {"agent_name": "fundamental_analysis",
                    "signal": "neutral", "confidence": 0.0},
                {"agent_name": "sentiment_analysis",
                    "signal": "neutral", "confidence": 0.0},
                {"agent_name": "valuation_analysis",
                    "signal": "neutral", "confidence": 0.0},
                {"agent_name": "risk_management",
                    "signal": "hold", "confidence": 1.0},
                {"agent_name": "macro_analyst_agent",
                    "signal": "neutral", "confidence": 0.0},
                {"agent_name": "market_wide_news_summary(沪深300指数)",
                    "signal": macro_news_payload["signal"], "confidence": macro_news_payload["confidence"]}
            ],
            "reasoning": "LLM API error. Defaulting to conservative hold based on risk management."
        })

    final_decision_message = HumanMessage(
        content=llm_response_content,
        name=agent_name,
    )

    if show_reasoning_flag:
        show_agent_reasoning(
            agent_name, f"Final LLM decision JSON: {llm_response_content}")

    agent_decision_details_value = {}
    try:
        decision_json = _parse_decision_json(llm_response_content)
        agent_decision_details_value = {
            "action": decision_json.get("action"),
            "quantity": decision_json.get("quantity"),
            "confidence": decision_json.get("confidence"),
            "reasoning_snippet": decision_json.get("reasoning", "")[:150] + "..."
        }
        logger.info(
            "📈 LLM 决策: action=%s quantity=%s confidence=%s",
            agent_decision_details_value["action"],
            agent_decision_details_value["quantity"],
            agent_decision_details_value["confidence"],
        )
    except (json.JSONDecodeError, ValueError) as parse_err:
        agent_decision_details_value = {
            "error": "Failed to parse LLM decision JSON from portfolio manager",
            "raw_response_snippet": llm_response_content[:200] + "..."
        }
        logger.error("⚠️ 无法解析投资组合 LLM 决策 JSON: %s", parse_err)

    show_workflow_status(f"{agent_name}: --- Portfolio Manager Completed ---")

    # The portfolio_management_agent is a terminal or near-terminal node in terms of new message generation for the main state.
    # It should return its own decision, and an updated state["messages"] that includes its decision.
    # As it's a汇聚点, it should ideally start with a cleaned list of messages from its inputs.
    # The cleaned_messages_for_processing already did this. We append its new message to this cleaned list.

    # If we strictly want to follow the pattern of `state["messages"] + [new_message]` for all non-leaf nodes,
    # then the `cleaned_messages_for_processing` should become the new `state["messages"]` for this node's context.
    # However, for simplicity and robustness, let's assume its output `messages` should just be its own message added to the cleaned input it processed.

    final_messages_output = cleaned_messages_for_processing + [final_decision_message]
    # Alternative if we want to be super strict about adding to the raw incoming state["messages"]:
    # final_messages_output = state["messages"] + [final_decision_message]
    # But this ^ is prone to the duplication we are trying to solve if not careful.
    # The most robust is that portfolio_manager provides its clear output, and the graph handles accumulation if needed for further steps (none in this case as it's END).

    # logger.info(
    # f"--- DEBUG: {agent_name} RETURN messages: {[msg.name for msg in final_messages_output]} ---")

    return {
        "messages": final_messages_output,
        "data": data,
        "metadata": {
            **state["metadata"],
            f"{agent_name}_decision_details": agent_decision_details_value,
            "agent_reasoning": llm_response_content
        }
    }


def allocate_positions(positions, available_capital=500000.0, prev_closes=None):
    """
    Calculate volumes for a list of positions using PositionSizer.

    Parameters
    ----------
    positions : list[dict]
        Each dict must have: symbol, symbol_name, C_mixed, prev_close.
    available_capital : float
        Total available capital.
    prev_closes : dict, optional
        {symbol: close_price} dict.

    Returns
    -------
    list[dict]
        Same list with weight, amount, volume added.
    """
    from src.execution.position_sizer import PositionSizer
    sizer = PositionSizer()
    return sizer.allocate(positions, available_capital, prev_closes)


def format_decision(action: str, quantity: int, confidence: float, agent_signals: list, reasoning: str, market_wide_news_summary: str = "未提供") -> dict:
    """Format the trading decision into a standardized output format.
    Think in English but output analysis in Chinese."""

    fundamental_signal = next(
        (s for s in agent_signals if s["agent_name"] == "fundamental_analysis"), None)
    valuation_signal = next(
        (s for s in agent_signals if s["agent_name"] == "valuation_analysis"), None)
    technical_signal = next(
        (s for s in agent_signals if s["agent_name"] == "technical_analysis"), None)
    sentiment_signal = next(
        (s for s in agent_signals if s["agent_name"] == "sentiment_analysis"), None)
    risk_signal = next(
        (s for s in agent_signals if s["agent_name"] == "risk_management"), None)
    # Existing macro signal from macro_analyst_agent (tool-based)
    general_macro_signal = next(
        (s for s in agent_signals if s["agent_name"] == "macro_analyst_agent"), None)
    # New market-wide news summary signal from macro_news_agent
    market_wide_news_signal = next(
        (s for s in agent_signals if s["agent_name"] == "market_wide_news_summary(沪深300指数)"), None)

    def signal_to_chinese(signal_data):
        if not signal_data:
            return "无数据"
        if signal_data.get("signal") == "bullish":
            return "看多"
        if signal_data.get("signal") == "bearish":
            return "看空"
        return "中性"

    detailed_analysis = f"""
====================================
          投资分析报告
====================================

一、策略分析

1. 基本面分析 (权重30%):
   信号: {signal_to_chinese(fundamental_signal)}
   置信度: {fundamental_signal['confidence']*100:.0f if fundamental_signal else 0}%
   要点:
   - 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}

2. 估值分析 (权重35%):
   信号: {signal_to_chinese(valuation_signal)}
   置信度: {valuation_signal['confidence']*100:.0f if valuation_signal else 0}%
   要点:
   - DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}
   - 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}

3. 技术分析 (权重25%):
   信号: {signal_to_chinese(technical_signal)}
   置信度: {technical_signal['confidence']*100:.0f if technical_signal else 0}%
   要点:
   - 趋势跟踪: ADX={technical_signal.get('strategy_signals', {}).get('trend_following', {}).get('metrics', {}).get('adx', 0.0):.2f if technical_signal else 0.0:.2f}
   - 均值回归: RSI(14)={technical_signal.get('strategy_signals', {}).get('mean_reversion', {}).get('metrics', {}).get('rsi_14', 0.0):.2f if technical_signal else 0.0:.2f}
   - 动量指标:
     * 1月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_1m', 0.0):.2% if technical_signal else 0.0:.2%}
     * 3月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_3m', 0.0):.2% if technical_signal else 0.0:.2%}
     * 6月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_6m', 0.0):.2% if technical_signal else 0.0:.2%}
   - 波动性: {technical_signal.get('strategy_signals', {}).get('volatility', {}).get('metrics', {}).get('historical_volatility', 0.0):.2% if technical_signal else 0.0:.2%}

4. 宏观分析 (综合权重15%):
   a) 常规宏观分析 (来自 Macro Analyst Agent):
      信号: {signal_to_chinese(general_macro_signal)}
      置信度: {general_macro_signal['confidence']*100:.0f if general_macro_signal else 0}%
      宏观环境: {general_macro_signal.get(
          'macro_environment', '无数据') if general_macro_signal else '无数据'}
      对股票影响: {general_macro_signal.get(
          'impact_on_stock', '无数据') if general_macro_signal else '无数据'}
      关键因素: {', '.join(general_macro_signal.get(
          'key_factors', ['无数据']) if general_macro_signal else ['无数据'])}

   b) 大盘宏观新闻分析 (来自 Macro News Agent):
      信号: {signal_to_chinese(market_wide_news_signal)}
      置信度: {market_wide_news_signal['confidence']*100:.0f if market_wide_news_signal else 0}%
      摘要或结论: {market_wide_news_signal.get(
          'reasoning', market_wide_news_summary) if market_wide_news_signal else market_wide_news_summary}

5. 情绪分析 (权重10%):
   信号: {signal_to_chinese(sentiment_signal)}
   置信度: {sentiment_signal['confidence']*100:.0f if sentiment_signal else 0}%
   分析: {sentiment_signal.get('reasoning', '无详细分析')
                             if sentiment_signal else '无详细分析'}

二、风险评估
风险评分: {risk_signal.get('risk_score', '无数据') if risk_signal else '无数据'}/10
主要指标:
- 波动率: {risk_signal.get('risk_metrics', {}).get('volatility', 0.0)*100:.1f if risk_signal else 0.0}%
- 最大回撤: {risk_signal.get('risk_metrics', {}).get('max_drawdown', 0.0)*100:.1f if risk_signal else 0.0}%
- VaR(95%): {risk_signal.get('risk_metrics', {}).get('value_at_risk_95', 0.0)*100:.1f if risk_signal else 0.0}%
- 市场风险: {risk_signal.get('risk_metrics', {}).get('market_risk_score', '无数据') if risk_signal else '无数据'}/10

三、投资建议
操作建议: {'买入' if action == 'buy' else '卖出' if action == 'sell' else '持有'}
交易数量: {quantity}股
决策置信度: {confidence*100:.0f}%

四、决策依据
{reasoning}

===================================="""

    return {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "分析报告": detailed_analysis
    }
