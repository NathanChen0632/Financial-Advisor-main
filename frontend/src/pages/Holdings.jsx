import { useState } from "react";
import Header from "../components/Layout/Header";
import { useHoldings } from "../hooks/useHoldings";
import { useTickerSignals } from "../hooks/useTickerSignals";
import { Plus, Trash2 } from "lucide-react";

const ACTION_STYLE = {
  BUY:  "text-emerald-400 bg-emerald-400/10",
  SELL: "text-red-400 bg-red-400/10",
  HOLD: "text-blue-400 bg-blue-400/10",
};

// Suggestion-only users' paper portfolio. Each holding is matched to the
// advisory signal from ticker_signals (written by the daily job).
export default function Holdings() {
  const { holdings, loading, error, addHolding, removeHolding, refresh } = useHoldings();
  const { byTicker } = useTickerSignals();

  const [ticker, setTicker] = useState("");
  const [qty,    setQty]    = useState("");
  const [price,  setPrice]  = useState("");
  const [busy,   setBusy]   = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (!ticker || !qty || !price) return;
    setBusy(true);
    const ok = await addHolding({ ticker, qty, entry_price: price });
    setBusy(false);
    if (ok) { setTicker(""); setQty(""); setPrice(""); }
  }

  return (
    <div>
      <Header
        title="My Holdings"
        subtitle="Your paper portfolio — suggestions update daily"
        onRefresh={refresh}
        loading={loading}
      />

      {error && (
        <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {/* Add holding */}
      <form onSubmit={submit} className="flex flex-wrap items-end gap-3 mb-6 bg-surface-card border border-surface-border rounded-xl p-4">
        <Field label="Ticker"      value={ticker} onChange={setTicker} placeholder="AAPL" width="w-28" upper />
        <Field label="Shares"      value={qty}    onChange={setQty}    placeholder="10"   width="w-24" type="number" />
        <Field label="Entry price" value={price}  onChange={setPrice}  placeholder="150"  width="w-28" type="number" />
        <button
          type="submit"
          disabled={busy}
          className="flex items-center gap-2 bg-brand text-black font-semibold rounded-lg px-4 py-2 text-sm hover:brightness-110 transition-all disabled:opacity-50"
        >
          <Plus className="w-4 h-4" /> Add
        </button>
      </form>

      {/* Holdings table */}
      {loading ? (
        <div className="h-40 bg-surface-card rounded-xl animate-pulse" />
      ) : holdings.length ? (
        <div className="bg-surface-card border border-surface-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500 border-b border-surface-border">
              <tr>
                <th className="text-left  px-4 py-3">Ticker</th>
                <th className="text-right px-4 py-3">Shares</th>
                <th className="text-right px-4 py-3">Entry</th>
                <th className="text-right px-4 py-3">Last</th>
                <th className="text-right px-4 py-3">P&L</th>
                <th className="text-left  px-4 py-3">Suggestion</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {holdings.map(h => {
                const sig  = byTicker[h.ticker?.toUpperCase()];
                const last = sig?.price != null ? Number(sig.price) : null;
                const pnl  = last != null ? (last - h.entry_price) * h.qty : null;
                const pnlPct = last != null && h.entry_price
                  ? ((last - h.entry_price) / h.entry_price) * 100 : null;
                const action = sig?.action;
                return (
                  <tr key={h.id} className="border-b border-surface-border/50 last:border-0">
                    <td className="px-4 py-3 font-medium text-white">{h.ticker}</td>
                    <td className="px-4 py-3 text-right text-slate-300">{Number(h.qty)}</td>
                    <td className="px-4 py-3 text-right text-slate-300">${Number(h.entry_price).toFixed(2)}</td>
                    <td className="px-4 py-3 text-right text-slate-300">{last != null ? `$${last.toFixed(2)}` : "—"}</td>
                    <td className={`px-4 py-3 text-right ${pnl == null ? "text-slate-500" : pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {pnl == null ? "—" : `${pnl >= 0 ? "+" : ""}$${pnl.toFixed(2)} (${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(1)}%)`}
                    </td>
                    <td className="px-4 py-3">
                      {action ? (
                        <span className={`text-xs px-2 py-0.5 rounded-md font-medium ${ACTION_STYLE[action]}`} title={sig?.rationale || ""}>
                          {action}
                        </span>
                      ) : <span className="text-xs text-slate-600">pending</span>}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button onClick={() => removeHolding(h.id)} className="text-slate-600 hover:text-red-400 transition-colors">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-16 text-slate-500">
          <p>No holdings yet.</p>
          <p className="text-xs mt-1">Add a position above to get sell/hold suggestions.</p>
        </div>
      )}
    </div>
  );
}

function Field({ label, value, onChange, placeholder, width, type = "text", upper }) {
  return (
    <div>
      <label className="block text-xs text-slate-400 mb-1">{label}</label>
      <input
        type={type}
        step="any"
        value={value}
        onChange={e => onChange(upper ? e.target.value.toUpperCase() : e.target.value)}
        placeholder={placeholder}
        className={`${width} bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-brand/50`}
      />
    </div>
  );
}
