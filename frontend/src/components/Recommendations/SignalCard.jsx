import { ArrowUpCircle, ArrowDownCircle, Clock, MinusCircle } from "lucide-react";

const ACTION_CONFIG = {
  BUY:  { icon: ArrowUpCircle,   color: "text-emerald-400", bg: "bg-emerald-400/10", border: "border-emerald-400/30", label: "BUY"  },
  SELL: { icon: ArrowDownCircle, color: "text-red-400",     bg: "bg-red-400/10",     border: "border-red-400/30",     label: "SELL" },
  HOLD: { icon: Clock,           color: "text-blue-400",    bg: "bg-blue-400/10",    border: "border-blue-400/30",    label: "HOLD" },
  WAIT: { icon: MinusCircle,     color: "text-slate-500",   bg: "bg-slate-700/30",   border: "border-slate-700",      label: "WAIT" },
};

export default function SignalCard({ signal }) {
  const cfg = ACTION_CONFIG[signal.action] || ACTION_CONFIG.WAIT;
  const Icon = cfg.icon;
  const ts   = signal.created_at
    ? new Date(signal.created_at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : "—";

  return (
    <div className={`bg-surface-card rounded-xl p-4 border ${cfg.border} transition-all hover:bg-surface-hover`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`p-1.5 rounded-lg ${cfg.bg}`}>
            <Icon className={`w-4 h-4 ${cfg.color}`} />
          </div>
          <span className="font-mono font-bold text-white text-base">{signal.ticker}</span>
        </div>
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${cfg.bg} ${cfg.color}`}>
          {cfg.label}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs mb-3">
        <Stat label="Price"  value={`$${(signal.price || 0).toFixed(2)}`} />
        {signal.stop_price   && <Stat label="Stop"   value={`$${signal.stop_price.toFixed(2)}`}   color="text-red-400/80" />}
        {signal.target_price && <Stat label="Target" value={`$${signal.target_price.toFixed(2)}`} color="text-emerald-400/80" />}
        {signal.atr_pct      && <Stat label="ATR %"  value={`${(signal.atr_pct * 100).toFixed(2)}%`} />}
      </div>

      {signal.action === "SELL" && signal.pnl_pct != null && (
        <div className={`text-xs font-mono font-semibold ${signal.pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
          P&L: {signal.pnl_pct >= 0 ? "+" : ""}{signal.pnl_pct.toFixed(2)}%
          {signal.exit_reason && <span className="text-slate-500 font-normal ml-1">({signal.exit_reason})</span>}
        </div>
      )}

      <p className="text-xs text-slate-600 mt-2">{ts}</p>
    </div>
  );
}

function Stat({ label, value, color = "text-white" }) {
  return (
    <div>
      <p className="text-slate-500 mb-0.5">{label}</p>
      <p className={`font-mono ${color}`}>{value}</p>
    </div>
  );
}
