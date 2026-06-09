import { usePortfolio }   from "../hooks/usePortfolio";
import { useSignals }     from "../hooks/useSignals";
import { usePerformance } from "../hooks/usePerformance";
import Header             from "../components/Layout/Header";
import SignalCard         from "../components/Recommendations/SignalCard";
import PositionCard       from "../components/Portfolio/PositionCard";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, Zap, DollarSign, Award } from "lucide-react";

export default function Dashboard() {
  const { positions, totalPnl, totalValue, winRate, loading: pLoad } = usePortfolio();
  const { latestByTicker, loading: sLoad }                           = useSignals(20);
  const { snapshots, summary }                                        = usePerformance(30);

  const actionSignals = latestByTicker.filter(s => ["BUY", "SELL"].includes(s.action));
  const pnlPositive   = totalPnl >= 0;

  // Format equity chart data
  const chartData = snapshots.map(s => ({
    date:   s.snapshot_date,
    equity: s.equity,
  }));

  return (
    <div>
      <Header title="Dashboard" subtitle="Live overview of your DQN trading system" />

      {/* KPI strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard
          label="Portfolio Value"
          value={`$${totalValue.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          icon={DollarSign}
          color="text-brand"
          bg="bg-brand-dim"
        />
        <KpiCard
          label="Unrealized P&L"
          value={`${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(2)}`}
          icon={Activity}
          color={pnlPositive ? "text-emerald-400" : "text-red-400"}
          bg={pnlPositive ? "bg-emerald-400/10" : "bg-red-400/10"}
        />
        <KpiCard
          label="Open Positions"
          value={positions.length}
          icon={Zap}
          color="text-blue-400"
          bg="bg-blue-400/10"
        />
        <KpiCard
          label="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          icon={Award}
          color="text-amber-400"
          bg="bg-amber-400/10"
        />
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
              <span className="ml-2 text-xs bg-brand-dim text-brand px-2 py-0.5 rounded-full">
                {actionSignals.length}
              </span>
            )}
          </h2>
          {sLoad ? (
            <Skeleton rows={3} />
          ) : actionSignals.length ? (
            <div className="space-y-3">
              {actionSignals.slice(0, 4).map((s, i) => (
                <SignalCard key={i} signal={s} />
              ))}
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
