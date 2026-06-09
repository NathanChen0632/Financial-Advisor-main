export default function TradeHistoryTable({ trades }) {
  if (!trades.length) {
    return <p className="text-slate-500 text-sm text-center py-8">No completed trades yet.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-surface-border text-slate-500 text-xs uppercase tracking-wider">
            <th className="text-left pb-3 pr-4 font-medium">Ticker</th>
            <th className="text-right pb-3 pr-4 font-medium">Entry</th>
            <th className="text-right pb-3 pr-4 font-medium">Exit</th>
            <th className="text-right pb-3 pr-4 font-medium">Shares</th>
            <th className="text-right pb-3 pr-4 font-medium">P&L</th>
            <th className="text-right pb-3 pr-4 font-medium">P&L %</th>
            <th className="text-right pb-3 font-medium">Reason</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => {
            const isWin = t.pnl > 0;
            const sign  = t.pnl > 0 ? "+" : "";
            return (
              <tr
                key={i}
                className="border-b border-surface-border/50 hover:bg-surface-hover transition-colors"
              >
                <td className="py-3 pr-4 font-mono font-semibold text-white">{t.ticker}</td>
                <td className="py-3 pr-4 text-right font-mono text-slate-300">${(t.entry_price || 0).toFixed(2)}</td>
                <td className="py-3 pr-4 text-right font-mono text-slate-300">${(t.exit_price  || 0).toFixed(2)}</td>
                <td className="py-3 pr-4 text-right text-slate-400">{t.qty || "—"}</td>
                <td className={`py-3 pr-4 text-right font-mono font-semibold ${isWin ? "text-emerald-400" : "text-red-400"}`}>
                  {sign}${(t.pnl || 0).toFixed(2)}
                </td>
                <td className={`py-3 pr-4 text-right font-mono ${isWin ? "text-emerald-400" : "text-red-400"}`}>
                  {sign}{(t.pnl_pct || 0).toFixed(2)}%
                </td>
                <td className="py-3 text-right text-xs text-slate-500 capitalize">{t.exit_reason || "signal"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
