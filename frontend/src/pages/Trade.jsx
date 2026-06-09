import { useState } from "react";
import Header from "../components/Layout/Header";
import { useOrderRequests } from "../hooks/useOrderRequests";
import { useAuth } from "../contexts/AuthContext";
import { ArrowUpRight, ArrowDownRight, Lock } from "lucide-react";

// Trading is restricted to the master account. Non-master users only ever get
// suggestions, never the ability to place real Alpaca orders. RLS on
// order_requests enforces this server-side too — this is the UI guard.
function MasterOnly() {
  return (
    <div className="text-center py-20 text-slate-500">
      <Lock className="w-7 h-7 mx-auto mb-3 opacity-40" />
      <p className="text-white font-medium">Trading isn't available on this account</p>
      <p className="text-xs mt-1">Live trading is limited to the master account. You get AI suggestions instead.</p>
    </div>
  );
}

const STATUS_STYLE = {
  pending:   "text-yellow-400 bg-yellow-400/10",
  executing: "text-blue-400 bg-blue-400/10",
  filled:    "text-emerald-400 bg-emerald-400/10",
  rejected:  "text-orange-400 bg-orange-400/10",
  error:     "text-red-400 bg-red-400/10",
};

// Master-only. Queue a buy/sell; the running monitor executes it on Alpaca and
// starts tracking the ticker with the DQN strategy.
export default function Trade() {
  const { isMaster } = useAuth();
  if (!isMaster) return <MasterOnly />;
  return <TradeInner />;
}

function TradeInner() {
  const { orders, loading, error, submitOrder, refresh } = useOrderRequests();
  const [ticker, setTicker] = useState("");
  const [side,   setSide]   = useState("buy");
  const [qty,    setQty]    = useState("");
  const [busy,   setBusy]   = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (!ticker) return;
    setBusy(true);
    const ok = await submitOrder({ ticker, side, qty });
    setBusy(false);
    if (ok) { setTicker(""); setQty(""); }
  }

  return (
    <div>
      <Header title="Trade" subtitle="Place a live order — the monitor executes it on Alpaca" onRefresh={refresh} loading={loading} />

      {error && (
        <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-4 py-3">{error}</div>
      )}

      <div className="bg-surface-card border border-surface-border rounded-xl p-5 mb-6 max-w-md">
        <div className="flex gap-2 mb-4">
          {["buy", "sell"].map(s => (
            <button
              key={s}
              onClick={() => setSide(s)}
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-all ${
                side === s
                  ? s === "buy" ? "bg-emerald-500 text-black" : "bg-red-500 text-white"
                  : "bg-surface border border-surface-border text-slate-400 hover:text-white"
              }`}
            >
              {s === "buy" ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
              {s.toUpperCase()}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="space-y-3">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Ticker</label>
            <input
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              placeholder="AAPL"
              className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-brand/50"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Shares {side === "buy" && <span className="text-slate-600">(optional — blank = risk-based sizing)</span>}
            </label>
            <input
              type="number"
              step="any"
              value={qty}
              onChange={e => setQty(e.target.value)}
              placeholder={side === "buy" ? "auto" : "all"}
              disabled={side === "sell"}
              className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-brand/50 disabled:opacity-50"
            />
            {side === "sell" && <p className="text-[11px] text-slate-600 mt-1">A sell closes the entire position.</p>}
          </div>
          <button
            type="submit"
            disabled={busy || !ticker}
            className={`w-full rounded-lg px-3 py-2.5 text-sm font-semibold transition-all disabled:opacity-50 ${
              side === "buy" ? "bg-emerald-500 text-black hover:brightness-110" : "bg-red-500 text-white hover:brightness-110"
            }`}
          >
            {busy ? "Submitting…" : `Submit ${side} order`}
          </button>
        </form>
        <p className="text-[11px] text-slate-600 mt-3">
          Orders are queued and executed by the monitor on its next poll. The monitor must be running with <code className="text-slate-400">--alpaca</code>.
        </p>
      </div>

      {/* Recent orders */}
      <h2 className="text-sm font-semibold text-white mb-2">Recent orders</h2>
      {orders.length ? (
        <div className="bg-surface-card border border-surface-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500 border-b border-surface-border">
              <tr>
                <th className="text-left px-4 py-2.5">Time</th>
                <th className="text-left px-4 py-2.5">Side</th>
                <th className="text-left px-4 py-2.5">Ticker</th>
                <th className="text-right px-4 py-2.5">Qty</th>
                <th className="text-left px-4 py-2.5">Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.map(o => (
                <tr key={o.id} className="border-b border-surface-border/50 last:border-0">
                  <td className="px-4 py-2.5 text-slate-500 text-xs">{new Date(o.created_at).toLocaleString()}</td>
                  <td className={`px-4 py-2.5 font-medium ${o.side === "buy" ? "text-emerald-400" : "text-red-400"}`}>{o.side.toUpperCase()}</td>
                  <td className="px-4 py-2.5 text-white">{o.ticker}</td>
                  <td className="px-4 py-2.5 text-right text-slate-300">{o.qty ?? "auto"}</td>
                  <td className="px-4 py-2.5">
                    <span className={`text-xs px-2 py-0.5 rounded-md font-medium ${STATUS_STYLE[o.status] || "text-slate-400 bg-slate-400/10"}`} title={o.error || ""}>
                      {o.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-slate-500">No orders yet.</p>
      )}
    </div>
  );
}
