import { TrendingUp, TrendingDown, Minus } from "lucide-react";

export default function PositionCard({ position }) {
  const pnl    = position.unrealized_pnl   || 0;
  const pnlPct = position.unrealized_pnl_pct || 0;
  const isUp   = pnl > 0;
  const isFlat = pnl === 0;

  const Icon      = isFlat ? Minus : isUp ? TrendingUp : TrendingDown;
  const colorCls  = isFlat ? "text-slate-400" : isUp ? "text-emerald-400" : "text-red-400";
  const bgCls     = isFlat ? "bg-slate-800" : isUp ? "bg-emerald-400/10" : "bg-red-400/10";
  const sign      = pnl > 0 ? "+" : "";

  return (
    <div className="bg-surface-card border border-surface-border rounded-xl p-4 hover:border-slate-600 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div>
          <span className="text-white font-semibold text-base font-mono">{position.ticker}</span>
          <p className="text-xs text-slate-500 mt-0.5">{position.strategy_type || "DQN Equity"}</p>
        </div>
        <div className={`p-1.5 rounded-lg ${bgCls}`}>
          <Icon className={`w-4 h-4 ${colorCls}`} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <Stat label="Shares"    value={position.qty} />
        <Stat label="Entry"     value={`$${(position.entry_price   || 0).toFixed(2)}`} />
        <Stat label="Current"   value={`$${(position.current_price || 0).toFixed(2)}`} />
        <Stat label="Value"     value={`$${((position.market_value || 0)).toLocaleString()}`} />
      </div>

      <div className={`mt-3 pt-3 border-t border-surface-border flex justify-between items-center`}>
        <span className="text-xs text-slate-500">Unrealized P&L</span>
        <span className={`text-sm font-semibold font-mono ${colorCls}`}>
          {sign}${pnl.toFixed(2)} ({sign}{pnlPct.toFixed(2)}%)
        </span>
      </div>

      {(position.stop_price || position.target_price) && (
        <div className="mt-2 flex gap-3 text-xs">
          {position.stop_price   && <span className="text-red-400/70">Stop: ${position.stop_price.toFixed(2)}</span>}
          {position.target_price && <span className="text-emerald-400/70">Target: ${position.target_price.toFixed(2)}</span>}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div>
      <p className="text-slate-500 mb-0.5">{label}</p>
      <p className="text-white font-mono">{value}</p>
    </div>
  );
}
