import { usePortfolio }      from "../hooks/usePortfolio";
import Header                  from "../components/Layout/Header";
import PositionCard            from "../components/Portfolio/PositionCard";
import TradeHistoryTable       from "../components/Portfolio/TradeHistoryTable";

export default function Portfolio() {
  const { positions, trades, loading, error, totalPnl, totalValue, winRate, refresh } = usePortfolio();

  const pnlPositive = totalPnl >= 0;

  return (
    <div>
      <Header
        title="Portfolio"
        subtitle="Current positions and completed trade history"
        onRefresh={refresh}
        loading={loading}
      />

      {error && <ErrorBanner message={error} />}

      {/* Summary row */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <SummaryTile label="Total Value"  value={`$${totalValue.toLocaleString("en-US", { minimumFractionDigits: 2 })}`} />
        <SummaryTile
          label="Unrealized P&L"
          value={`${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(2)}`}
          color={pnlPositive ? "text-emerald-400" : "text-red-400"}
        />
        <SummaryTile label="Win Rate" value={`${winRate.toFixed(1)}%`} color="text-amber-400" />
      </div>

      {/* Open positions */}
      <Section title={`Open Positions (${positions.length})`}>
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map(i => <div key={i} className="h-44 bg-surface-hover rounded-xl animate-pulse" />)}
          </div>
        ) : positions.length ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {positions.map((p, i) => <PositionCard key={i} position={p} />)}
          </div>
        ) : (
          <p className="text-slate-500 text-sm text-center py-8">No open positions.</p>
        )}
      </Section>

      {/* Trade history */}
      <Section title={`Trade History (${trades.length})`} className="mt-6">
        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4].map(i => <div key={i} className="h-10 bg-surface-hover rounded animate-pulse" />)}
          </div>
        ) : (
          <TradeHistoryTable trades={trades} />
        )}
      </Section>
    </div>
  );
}

function SummaryTile({ label, value, color = "text-white" }) {
  return (
    <div className="bg-surface-card border border-surface-border rounded-xl px-4 py-3">
      <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider">{label}</p>
      <p className={`text-xl font-mono font-semibold ${color}`}>{value}</p>
    </div>
  );
}

function Section({ title, children, className = "" }) {
  return (
    <div className={`bg-surface-card border border-surface-border rounded-xl p-5 ${className}`}>
      <h2 className="text-sm font-medium text-white mb-4">{title}</h2>
      {children}
    </div>
  );
}

function ErrorBanner({ message }) {
  return (
    <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-4 py-3">
      Error loading portfolio: {message}
    </div>
  );
}
