import { useState }        from "react";
import { usePerformance }  from "../hooks/usePerformance";
import Header              from "../components/Layout/Header";
import MetricsGrid         from "../components/Performance/MetricsGrid";
import {
  AreaChart, Area,
  BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";

const WINDOWS = [
  { label: "30 days",  days: 30  },
  { label: "60 days",  days: 60  },
  { label: "90 days",  days: 90  },
  { label: "180 days", days: 180 },
];

export default function Performance() {
  const [window, setWindow]           = useState(90);
  const { snapshots, summary, loading, error, refresh } = usePerformance(window);

  const equityData = snapshots.map(s => ({
    date:   s.snapshot_date?.slice(5),   // MM-DD
    equity: s.equity,
    bah:    s.bah_equity,
  }));

  const returnData = snapshots.slice(1).map((s, i) => {
    const prev    = snapshots[i];
    const dailyRet = prev ? ((s.equity - prev.equity) / prev.equity) * 100 : 0;
    return { date: s.snapshot_date?.slice(5), return: parseFloat(dailyRet.toFixed(3)) };
  });

  return (
    <div>
      <Header
        title="Performance"
        subtitle="Equity curve, daily returns, and risk metrics"
        onRefresh={refresh}
        loading={loading}
      />

      {error && (
        <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {/* Window selector */}
      <div className="flex gap-2 mb-5">
        {WINDOWS.map(w => (
          <button
            key={w.days}
            onClick={() => setWindow(w.days)}
            className={`text-xs px-3 py-1.5 rounded-lg transition-all ${
              window === w.days
                ? "bg-brand text-black font-semibold"
                : "bg-surface-card border border-surface-border text-slate-400 hover:text-white"
            }`}
          >
            {w.label}
          </button>
        ))}
      </div>

      <MetricsGrid summary={summary} />

      {loading ? (
        <div className="space-y-4">
          <div className="h-52 bg-surface-card rounded-xl animate-pulse" />
          <div className="h-40 bg-surface-card rounded-xl animate-pulse" />
        </div>
      ) : snapshots.length > 1 ? (
        <>
          {/* Equity vs Buy-and-Hold */}
          <div className="bg-surface-card border border-surface-border rounded-xl p-5 mb-4">
            <h2 className="text-sm font-medium text-white mb-4">Equity Curve vs Buy &amp; Hold</h2>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={equityData}>
                <defs>
                  <linearGradient id="strat" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00d4aa" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#00d4aa" stopOpacity={0}    />
                  </linearGradient>
                  <linearGradient id="bah" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#f97316" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0}    />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1e2740" strokeDasharray="4 4" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} tickLine={false} axisLine={false}
                       tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} width={50} />
                <Tooltip
                  contentStyle={{ background: "#161b27", border: "1px solid #1e2740", borderRadius: 8 }}
                  labelStyle={{ color: "#94a3b8", fontSize: 10 }}
                  formatter={(v, name) => [`$${v.toLocaleString()}`, name === "equity" ? "DQN Strategy" : "Buy & Hold"]}
                />
                <Area type="monotone" dataKey="equity" stroke="#00d4aa" strokeWidth={2} fill="url(#strat)" dot={false} name="equity" />
                {equityData.some(d => d.bah) && (
                  <Area type="monotone" dataKey="bah" stroke="#f97316" strokeWidth={1.5} fill="url(#bah)" dot={false} name="bah" strokeDasharray="4 4" />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Daily returns bar chart */}
          <div className="bg-surface-card border border-surface-border rounded-xl p-5">
            <h2 className="text-sm font-medium text-white mb-4">Daily Returns (%)</h2>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={returnData} barCategoryGap="20%">
                <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} tickLine={false} axisLine={false}
                       tickFormatter={v => `${v}%`} width={40} />
                <Tooltip
                  contentStyle={{ background: "#161b27", border: "1px solid #1e2740", borderRadius: 8 }}
                  labelStyle={{ color: "#94a3b8", fontSize: 10 }}
                  formatter={v => [`${v}%`, "Return"]}
                />
                <ReferenceLine y={0} stroke="#1e2740" />
                <Bar
                  dataKey="return"
                  radius={[2, 2, 0, 0]}
                  fill="#00d4aa"
                  // Negative bars rendered in red via Cell would need extra work;
                  // using a single green tint keeps the chart simple and clean.
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <div className="bg-surface-card border border-surface-border rounded-xl p-10 text-center text-slate-500 text-sm">
          No performance data yet. Once your Python monitor starts writing snapshots, the charts will appear here.
        </div>
      )}
    </div>
  );
}
