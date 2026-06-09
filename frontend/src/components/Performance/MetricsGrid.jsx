import { TrendingUp, TrendingDown, BarChart2, Activity } from "lucide-react";

export default function MetricsGrid({ summary }) {
  if (!summary) return null;

  const totalReturnNum = parseFloat(summary.totalReturn);
  const isUp = totalReturnNum >= 0;

  const metrics = [
    {
      label: "Current Equity",
      value: `$${(summary.currentEquity || 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      icon: Activity,
      color: "text-brand",
      bg: "bg-brand-dim",
    },
    {
      label: "Total Return",
      value: `${isUp ? "+" : ""}${summary.totalReturn}%`,
      icon: isUp ? TrendingUp : TrendingDown,
      color: isUp ? "text-emerald-400" : "text-red-400",
      bg: isUp ? "bg-emerald-400/10" : "bg-red-400/10",
    },
    {
      label: "Sharpe Ratio",
      value: summary.sharpe,
      icon: BarChart2,
      color: "text-blue-400",
      bg: "bg-blue-400/10",
      hint: "> 1.0 is good",
    },
    {
      label: "Max Drawdown",
      value: `${summary.maxDrawdown}%`,
      icon: TrendingDown,
      color: "text-amber-400",
      bg: "bg-amber-400/10",
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {metrics.map(({ label, value, icon: Icon, color, bg, hint }) => (
        <div key={label} className="bg-surface-card border border-surface-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
            <div className={`p-1.5 rounded-lg ${bg}`}>
              <Icon className={`w-3.5 h-3.5 ${color}`} />
            </div>
          </div>
          <p className={`text-2xl font-mono font-semibold ${color}`}>{value}</p>
          {hint && <p className="text-xs text-slate-600 mt-1">{hint}</p>}
        </div>
      ))}
    </div>
  );
}
