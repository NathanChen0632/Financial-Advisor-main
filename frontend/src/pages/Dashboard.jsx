import { Link }            from "react-router-dom";
import { useAuth }         from "../contexts/AuthContext";
import { usePortfolio }    from "../hooks/usePortfolio";
import { useSignals }      from "../hooks/useSignals";
import { usePerformance }  from "../hooks/usePerformance";
import { useHoldings }     from "../hooks/useHoldings";
import { useWatchlist }    from "../hooks/useWatchlist";
import { useTickerSignals } from "../hooks/useTickerSignals";
import { useRecommendations } from "../hooks/useRecommendations";
import Header             from "../components/Layout/Header";
import SignalCard         from "../components/Recommendations/SignalCard";
import PositionCard       from "../components/Portfolio/PositionCard";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, Zap, DollarSign, Award, Wallet, Eye, ArrowUpCircle, ArrowDownCircle, Clock } from "lucide-react";

// Role decides what "your dashboard" means:
//  - master: the live Alpaca account (portfolio, equity, system signals)
//  - user:   their OWN holdings + watchlist + shared AI insights — never the
//            master's account. Each regular user only sees their own data.
export default function Dashboard() {
  const { isMaster } = useAuth();
  return isMaster ? <MasterDashboard /> : <UserDashboard />;
}

/* ============================ MASTER ============================ */

function MasterDashboard() {
  const { positions, totalPnl, totalValue, winRate, loading: pLoad } = usePortfolio();
  const { latestByTicker, loading: sLoad }                           = useSignals(20);
  const { snapshots }                                                = usePerformance(30);

  const actionSignals = latestByTicker.filter(s => ["BUY", "SELL"].includes(s.action));
  const pnlPositive   = totalPnl >= 0;
  const chartData = snapshots.map(s => ({ date: s.snapshot_date, equity: s.equity }));

  return (
    <div>
      <Header title="Dashboard" subtitle="Live overview of your DQN trading system" />

      {/* KPI strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard label="Portfolio Value" value={`$${totalValue.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} icon={DollarSign} color="text-brand" bg="bg-brand-dim" />
        <KpiCard label="Unrealized P&L" value={`${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(2)}`} icon={Activity} color={pnlPositive ? "text-emerald-400" : "text-red-400"} bg={pnlPositive ? "bg-emerald-400/10" : "bg-red-400/10"} />
        <KpiCard label="Open Positions" value={positions.length} icon={Zap} color="text-blue-400" bg="bg-blue-400/10" />
        <KpiCard label="Win Rate" value={`${winRate.toFixed(1)}%`} icon={Award} color="text-amber-400" bg="bg-amber-400/10" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Equity curve mini */}
        <div className="lg:col-span-2 bg-surface-card border border-surface-border rounded-xl p-5">
          <h2 className="text-sm font-medium text-white mb-4">Equity — Last 30 Days</h2>
          {chartData.length > 1 ? (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00d4aa" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#00d4aa" stopOpacity={0}    />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickLine={false} axisLine={false}
                       tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} width={48} />
                <Tooltip
                  contentStyle={{ background: "#161b27", border: "1px solid #1e2740", borderRadius: 8 }}
                  labelStyle={{ color: "#94a3b8", fontSize: 11 }}
                  itemStyle={{ color: "#00d4aa", fontSize: 12 }}
                  formatter={v => [`$${v.toLocaleString()}`, "Equity"]}
                />
                <Area type="monotone" dataKey="equity" stroke="#00d4aa" strokeWidth={2} fill="url(#eq)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart />
          )}
        </div>

        {/* Latest actionable signals */}
        <div className="bg-surface-card border border-surface-border rounded-xl p-5">
          <h2 className="text-sm font-medium text-white mb-4">
            Active Signals
            {actionSignals.length > 0 && (
              <span className="ml-2 text-xs bg-brand-dim text-brand px-2 py-0.5 rounded-full">{actionSignals.length}</span>
            )}
          </h2>
          {sLoad ? (
            <Skeleton rows={3} />
          ) : actionSignals.length ? (
            <div className="space-y-3">
              {actionSignals.slice(0, 4).map((s, i) => <SignalCard key={i} signal={s} />)}
            </div>
          ) : (
            <p className="text-slate-500 text-sm text-center py-6">No active signals right now.</p>
          )}
        </div>
      </div>

      {/* Open positions preview */}
      {positions.length > 0 && (
        <div className="mt-6">
          <h2 className="text-sm font-medium text-white mb-3">Open Positions</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {positions.slice(0, 3).map((p, i) => <PositionCard key={i} position={p} />)}
          </div>
        </div>
      )}
    </div>
  );
}

/* ============================= USER ============================= */

const ADVISORY = {
  BUY:  { icon: ArrowUpCircle,   color: "text-emerald-400", bg: "bg-emerald-400/10" },
  SELL: { icon: ArrowDownCircle, color: "text-red-400",     bg: "bg-red-400/10"     },
  HOLD: { icon: Clock,           color: "text-blue-400",    bg: "bg-blue-400/10"    },
};

function UserDashboard() {
  const { user }                  = useAuth();
  const { holdings, loading: hLoad } = useHoldings();
  const { items: watchlist }      = useWatchlist();
  const { byTicker }              = useTickerSignals();
  const { recs }                  = useRecommendations();

  // Mark-to-market the user's own holdings using the daily signal price.
  let costValue = 0, marketValue = 0, priced = false;
  for (const h of holdings) {
    const sig  = byTicker[h.ticker?.toUpperCase()];
    const last = sig?.price != null ? Number(sig.price) : null;
    costValue   += Number(h.entry_price) * Number(h.qty);
    marketValue += (last != null ? last : Number(h.entry_price)) * Number(h.qty);
    if (last != null) priced = true;
  }
  const pnl    = priced ? marketValue - costValue : null;
  const pnlPct = priced && costValue ? (pnl / costValue) * 100 : null;

  // Actionable suggestions across everything the user tracks.
  const tracked = [...new Set([
    ...holdings.map(h => h.ticker?.toUpperCase()),
    ...watchlist.map(w => w.ticker?.toUpperCase()),
  ].filter(Boolean))];
  const actionable = tracked
    .map(t => byTicker[t])
    .filter(s => s && (s.action === "BUY" || s.action === "SELL"));

  return (
    <div>
      <Header title="Dashboard" subtitle={`Signed in as ${user?.email} · your watchlist signals & AI insights`} />

      {/* KPI strip — entirely the user's own data */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard label="My Holdings" value={holdings.length} icon={Wallet} color="text-brand" bg="bg-brand-dim" />
        <KpiCard
          label="Holdings P&L"
          value={pnl == null ? "—" : `${pnl >= 0 ? "+" : ""}$${pnl.toFixed(2)}`}
          icon={Activity}
          color={pnl == null ? "text-slate-400" : pnl >= 0 ? "text-emerald-400" : "text-red-400"}
          bg={pnl == null ? "bg-slate-700/30" : pnl >= 0 ? "bg-emerald-400/10" : "bg-red-400/10"}
        />
        <KpiCard label="Watchlist" value={watchlist.length} icon={Eye} color="text-blue-400" bg="bg-blue-400/10" />
        <KpiCard label="Active Signals" value={actionable.length} icon={Zap} color="text-amber-400" bg="bg-amber-400/10" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Holdings suggestions */}
        <div className="bg-surface-card border border-surface-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium text-white">Your Holdings — suggestions</h2>
            <Link to="/holdings" className="text-xs text-brand hover:underline">Manage</Link>
          </div>
          {hLoad ? (
            <Skeleton rows={3} />
          ) : holdings.length ? (
            <div className="space-y-2">
              {holdings.slice(0, 6).map(h => <AdvisoryRow key={h.id} ticker={h.ticker} sig={byTicker[h.ticker?.toUpperCase()]} sub={`${Number(h.qty)} sh @ $${Number(h.entry_price).toFixed(2)}`} />)}
            </div>
          ) : (
            <EmptyCta to="/holdings" label="Add your first holding" hint="Upload positions to get sell/hold suggestions." />
          )}
        </div>

        {/* Watchlist buy signals */}
        <div className="bg-surface-card border border-surface-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium text-white">Watchlist — buy signals</h2>
            <Link to="/watchlist" className="text-xs text-brand hover:underline">Manage</Link>
          </div>
          {watchlist.length ? (
            <div className="space-y-2">
              {watchlist.slice(0, 6).map(w => <AdvisoryRow key={w.id} ticker={w.ticker} sig={byTicker[w.ticker?.toUpperCase()]} />)}
            </div>
          ) : (
            <EmptyCta to="/watchlist" label="Build your watchlist" hint="Add tickers to track buy/hold signals." />
          )}
        </div>
      </div>

      {/* AI market recommendations (shared, read-only) */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-white">AI Market Picks</h2>
          <Link to="/recommendations" className="text-xs text-brand hover:underline">See all</Link>
        </div>
        {recs.length ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {recs.slice(0, 3).map((r, i) => (
              <SignalCard key={i} signal={{ ticker: r.ticker, action: "BUY", price: r.price ?? 0 }} />
            ))}
          </div>
        ) : (
          <p className="text-slate-500 text-sm">No AI picks yet — they refresh when the daily job runs.</p>
        )}
      </div>
    </div>
  );
}

function AdvisoryRow({ ticker, sig, sub }) {
  const action = sig?.action;
  const cfg = ADVISORY[action];
  const Icon = cfg?.icon;
  return (
    <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-surface hover:bg-surface-hover transition-colors">
      <div className="flex items-center gap-2 min-w-0">
        <span className="font-mono font-semibold text-white">{ticker}</span>
        {sub && <span className="text-xs text-slate-500 truncate">{sub}</span>}
      </div>
      {cfg ? (
        <span className={`flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full ${cfg.bg} ${cfg.color}`} title={sig?.rationale || ""}>
          <Icon className="w-3.5 h-3.5" /> {action}
        </span>
      ) : (
        <span className="text-xs text-slate-600">pending</span>
      )}
    </div>
  );
}

function EmptyCta({ to, label, hint }) {
  return (
    <div className="text-center py-8">
      <p className="text-sm text-slate-400">{hint}</p>
      <Link to={to} className="inline-block mt-3 text-sm font-semibold bg-brand text-black rounded-lg px-4 py-2 hover:brightness-110 transition-all">
        {label}
      </Link>
    </div>
  );
}

/* ============================ SHARED ============================ */

function KpiCard({ label, value, icon: Icon, color, bg }) {
  return (
    <div className="bg-surface-card border border-surface-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
        <div className={`p-1.5 rounded-lg ${bg}`}>
          <Icon className={`w-3.5 h-3.5 ${color}`} />
        </div>
      </div>
      <p className={`text-2xl font-mono font-semibold ${color}`}>{value}</p>
    </div>
  );
}

function EmptyChart() {
  return (
    <div className="h-44 flex items-center justify-center text-slate-600 text-sm">
      No performance data yet — connect Supabase to see your equity curve.
    </div>
  );
}

function Skeleton({ rows = 3 }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-20 bg-surface-hover rounded-xl animate-pulse" />
      ))}
    </div>
  );
}
