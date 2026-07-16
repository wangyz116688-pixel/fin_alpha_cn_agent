"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000/api/demo";

type Metric = { key: string; label: string; days: number; total_return: number; annual_return: number; max_drawdown: number; sharpe: number; sortino: number; calmar: number; win_rate: number; profit_loss_ratio: number };
type Point = { date: string; value: number; return: number };
type Position = { symbol: string; name: string; shares: number; previous_close: number | null; amount: number | null; weight: number | null; confidence: number | null; factors: { name: string; percentile: number }[]; reason: string };
type Recommendation = { date: string; requested_date: string; signal_date: string; adjusted: boolean; adjustment_note?: string; source_label: string; available_capital: number; invested_amount: number; cash: number; positions: Position[]; risk_notice: string };
type Overview = { available_range: { start: string; end: string }; metrics: Metric[]; default_period: string; equity_curve: Point[]; quick_questions: string[]; strategy: Record<string, string | number> };
type Message = { id: string; role: "assistant" | "user"; text: string; kind?: "welcome" | "recommendation" | "metrics" | "strategy" | "progress" | "error"; result?: Recommendation; progress?: number; stage?: string };

const welcome: Message = { id: "welcome", role: "assistant", kind: "welcome", text: "你好，我是 AlphaAgent。告诉我一个日期，我会从正式回测缓存中还原当日的三只持仓、仓位与入选依据。" };
const money = new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 0 });
const pct = (v?: number | null) => v == null ? "—" : `${(v * 100).toFixed(2)}%`;
const num = (v?: number | null) => v == null ? "—" : v.toFixed(2);

function Icon({ name }: { name: "send" | "calendar" | "spark" | "chart" | "shield" | "chevron" }) {
  const paths = {
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    calendar: <><path d="M8 2v4M16 2v4M3 10h18"/><rect x="3" y="4" width="18" height="18" rx="2"/></>,
    spark: <><path d="m12 3 1.3 4.1L17 9l-3.7 1.9L12 15l-1.3-4.1L7 9l3.7-1.9Z"/><path d="m19 15 .7 2.3L22 18.5l-2.3 1.2L19 22l-.7-2.3-2.3-1.2 2.3-1.2Z"/></>,
    chart: <><path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/></>,
    shield: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
  };
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

function EquityChart({ points, highlight }: { points: Point[]; highlight?: string }) {
  const width = 640, height = 220, pad = 14;
  const values = points.map(p => p.return);
  const min = Math.min(...values, 0), max = Math.max(...values, 0);
  const x = (i: number) => pad + i / Math.max(1, points.length - 1) * (width - pad * 2);
  const y = (v: number) => height - pad - (v - min) / Math.max(.0001, max - min) * (height - pad * 2);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.return).toFixed(1)}`).join(" ");
  const area = `${path} L${x(points.length - 1)},${height - pad} L${pad},${height - pad} Z`;
  let hi = highlight ? points.findIndex(p => p.date === highlight) : -1;
  if (hi < 0 && highlight) hi = points.findLastIndex(p => p.date <= highlight);
  return <div className="chart-wrap">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="2026上半年策略净值曲线">
      <defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#38d59f" stopOpacity=".26"/><stop offset="1" stopColor="#38d59f" stopOpacity="0"/></linearGradient></defs>
      {[.25,.5,.75].map(t => <line key={t} x1={pad} x2={width-pad} y1={pad+t*(height-pad*2)} y2={pad+t*(height-pad*2)} className="grid-line" />)}
      <path d={area} fill="url(#area)"/><path d={path} className="equity-line"/>
      {hi >= 0 && <><line x1={x(hi)} x2={x(hi)} y1={pad} y2={height-pad} className="highlight-line"/><circle cx={x(hi)} cy={y(points[hi].return)} r="5" className="highlight-dot"/></>}
    </svg>
    <div className="chart-axis"><span>01/06</span><span>03/31</span><span>06/30</span></div>
  </div>;
}

function PositionCard({ item, index }: { item: Position; index: number }) {
  return <details className="position-card">
    <summary>
      <div className="rank">0{index + 1}</div>
      <div className="stock"><strong>{item.name}</strong><span>{item.symbol}</span></div>
      <div className="position-main"><strong>{pct(item.weight)}</strong><span>建议仓位</span></div>
      <div className="expand"><span>查看依据</span><Icon name="chevron" /></div>
    </summary>
    <div className="position-detail">
      <div className="trade-grid"><span>建议股数<strong>{item.shares.toLocaleString()} 股</strong></span><span>前收<strong>{item.previous_close == null ? "—" : `¥${item.previous_close.toFixed(2)}`}</strong></span><span>买入金额<strong>{item.amount == null ? "—" : money.format(item.amount)}</strong></span><span>置信度<strong>{pct(item.confidence)}</strong></span></div>
      <p>{item.reason}</p>
      {!!item.factors.length && <div className="factor-list">{item.factors.map(f => <span key={f.name}>{f.name}<b>{Math.round(f.percentile * 100)}分位</b></span>)}</div>}
    </div>
  </details>;
}

export default function Home() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [messages, setMessages] = useState<Message[]>([welcome]);
  const [input, setInput] = useState("");
  const [dateValue, setDateValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [backendError, setBackendError] = useState("");
  const [period, setPeriod] = useState("2026_h1");
  const [highlight, setHighlight] = useState<string>();
  const endRef = useRef<HTMLDivElement>(null);

  const loadOverview = async () => {
    setBackendError("");
    try {
      const response = await fetch(`${API_BASE}/overview`);
      if (!response.ok) throw new Error("后端未响应");
      const body = await response.json();
      setOverview(body.data); setPeriod(body.data.default_period);
    } catch {
      setBackendError("未连接到本地策略服务。请先运行项目根目录的一键启动脚本。");
    }
  };

  useEffect(() => { loadOverview(); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  const currentMetric = useMemo(() => overview?.metrics.find(m => m.key === period), [overview, period]);

  const poll = async (jobId: string, messageId: string) => {
    for (let n = 0; n < 360; n++) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const response = await fetch(`${API_BASE}/jobs/${jobId}`);
      const body = await response.json(); const job = body.data;
      if (job.status === "completed") {
        setMessages(list => list.map(m => m.id === messageId ? { ...m, kind: "recommendation", text: `${job.result.date} 交易建议已生成。`, result: job.result } : m));
        setHighlight(job.result.date); setBusy(false); return;
      }
      if (job.status === "failed") throw new Error(job.error || "策略计算失败");
      setMessages(list => list.map(m => m.id === messageId ? { ...m, stage: job.stage, progress: job.progress } : m));
    }
    throw new Error("计算时间超出预期，请稍后重试。");
  };

  const send = async (text = input) => {
    const message = text.trim(); if (!message || busy) return;
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", text: message };
    setMessages(list => [...list, userMessage]); setInput(""); setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, date: dateValue || undefined }) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "请求失败");
      const data = body.data; const id = crypto.randomUUID();
      if (data.status === "pending") {
        setMessages(list => [...list, { id, role: "assistant", kind: "progress", text: data.reply, stage: data.stage, progress: data.progress }]);
        await poll(data.job_id, id); return;
      }
      if (data.intent === "advice") { setMessages(list => [...list, { id, role: "assistant", kind: "recommendation", text: data.reply, result: data.result }]); setHighlight(data.result.date); }
      else if (data.intent === "metrics") { setMessages(list => [...list, { id, role: "assistant", kind: "metrics", text: data.reply }]); setPeriod("2026_h1"); }
      else setMessages(list => [...list, { id, role: "assistant", kind: data.intent === "strategy" ? "strategy" : undefined, text: data.reply }]);
      setBusy(false);
    } catch (error) {
      setMessages(list => [...list, { id: crypto.randomUUID(), role: "assistant", kind: "error", text: error instanceof Error ? error.message : "服务暂时不可用，请重试。" }]); setBusy(false);
    }
  };

  const onSubmit = (e: FormEvent) => { e.preventDefault(); send(); };

  return <main className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark">A<span>α</span></div><div><strong>AlphaAgent</strong><small>QUANTITATIVE INTELLIGENCE</small></div></div>
      <div className="strategy-pill"><i></i><span>正式策略在线</span><b>IC BLEND · 3 POSITIONS</b></div>
    </header>

    <section className="workspace">
      <div className="chat-panel">
        <div className="chat-heading"><div><span className="eyebrow">INVESTMENT COPILOT</span><h1>策略对话台</h1></div><div className="range">可查询范围 <b>{overview ? `${overview.available_range.start} — ${overview.available_range.end}` : "加载中"}</b></div></div>
        {backendError && <div className="connection-error"><span>服务未连接</span><p>{backendError}</p><button onClick={loadOverview}>重新连接</button></div>}
        <div className="messages">
          {messages.map(message => <div key={message.id} className={`message-row ${message.role}`}>
            {message.role === "assistant" && <div className="avatar"><Icon name="spark" /></div>}
            <div className={`bubble ${message.kind || ""}`}>
              <p className="message-text">{message.text}</p>
              {message.kind === "welcome" && <div className="quick-grid">{(overview?.quick_questions || ["今天交易建议", "2026-06-30 的建议", "查看策略表现"]).map((q, i) => <button key={q} onClick={() => send(q)}><Icon name={i === 2 ? "chart" : i === 1 ? "calendar" : "spark"}/><span>{q}</span></button>)}</div>}
              {message.kind === "progress" && <div className="progress-card"><div><span>{message.stage}</span><b>{message.progress}%</b></div><div className="progress-track"><i style={{ width: `${message.progress}%` }}/></div><small>正在调用正式策略，请勿关闭页面</small></div>}
              {message.result && <div className="recommendation">
                <div className="rec-head"><div><span className="date-label">建议交易日</span><strong>{message.result.date}</strong></div><span className="source-tag">{message.result.source_label}</span></div>
                {message.result.adjustment_note && <div className="adjust-note">{message.result.adjustment_note}</div>}
                <div className="signal-note">信号数据截止 <b>{message.result.signal_date}</b><span>·</span> 可用资金 <b>{money.format(message.result.available_capital)}</b></div>
                <div className="positions">{message.result.positions.length ? message.result.positions.map((p, i) => <PositionCard item={p} index={i} key={p.symbol}/>) : <div className="empty-state">当日策略为空仓</div>}</div>
                <div className="allocation-strip"><div><span>计划投入</span><b>{money.format(message.result.invested_amount)}</b></div><div><span>剩余现金</span><b>{money.format(message.result.cash)}</b></div></div>
                <div className="risk"><Icon name="shield"/><span>{message.result.risk_notice}</span></div>
              </div>}
            </div>
          </div>)}
          {busy && !messages.some(m => m.kind === "progress") && <div className="typing"><i></i><i></i><i></i></div>}
          <div ref={endRef}/>
        </div>
        <form className="composer" onSubmit={onSubmit}>
          <label className="date-picker"><Icon name="calendar"/><input aria-label="选择建议日期" type="date" value={dateValue} max={new Date().toISOString().slice(0,10)} onChange={e => setDateValue(e.target.value)}/></label>
          <input aria-label="输入问题" value={input} onChange={e => setInput(e.target.value)} placeholder="输入日期或问题，例如：2026年6月30日交易建议是什么？"/>
          <button className="send-button" disabled={busy || !input.trim()} aria-label="发送" type="submit"><Icon name="send"/></button>
        </form>
        <div className="composer-note">Enter 发送 · 历史结果缓存优先 · 未缓存日期自动计算</div>
      </div>

      <aside className="insight-panel">
        <div className="aside-head"><div><span className="eyebrow">STRATEGY MONITOR</span><h2>策略表现</h2></div><span className="live"><i></i>LIVE</span></div>
        <div className="period-tabs">{overview?.metrics.map(m => <button className={period === m.key ? "active" : ""} onClick={() => setPeriod(m.key)} key={m.key}>{m.label}</button>)}</div>
        <div className="hero-metric"><span>累计收益</span><strong>{pct(currentMetric?.total_return)}</strong><small>{currentMetric?.days || "—"} 个收益日</small></div>
        <div className="metric-grid">
          <div><span>年化收益</span><b>{pct(currentMetric?.annual_return)}</b></div><div><span>最大回撤</span><b className="down">-{pct(currentMetric?.max_drawdown)}</b></div>
          <div><span>夏普比率</span><b>{num(currentMetric?.sharpe)}</b></div><div><span>索泰诺</span><b>{num(currentMetric?.sortino)}</b></div>
          <div><span>卡玛比率</span><b>{num(currentMetric?.calmar)}</b></div><div><span>日胜率</span><b>{pct(currentMetric?.win_rate)}</b></div>
          <div><span>日盈亏比</span><b>{num(currentMetric?.profit_loss_ratio)}</b></div><div><span>初始资金</span><b>¥50万</b></div>
        </div>
        <div className="chart-section"><div className="chart-title"><div><span>2026 H1</span><b>策略净值</b></div><small>{highlight ? `高亮 ${highlight}` : "2026-01-01 — 06-30"}</small></div>{overview?.equity_curve.length ? <EquityChart points={overview.equity_curve} highlight={highlight}/> : <div className="chart-skeleton"/>}</div>
        <div className="method-card"><div className="method-icon"><Icon name="spark"/></div><div><span>当前策略</span><strong>Train IC Blend</strong><p>500只固定种子股票池 · 得分配资 · 3只持仓</p></div></div>
        <p className="disclaimer">回测结果不代表未来表现，不构成投资建议。</p>
      </aside>
    </section>
  </main>;
}
