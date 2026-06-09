import { useState } from "react";
import Header from "../components/Layout/Header";
import { useWatchlist } from "../hooks/useWatchlist";
import { useTickerSignals } from "../hooks/useTickerSignals";
import { Plus, Trash2, Eye } from "lucide-react";

const ACTION_STYLE = {
  BUY:  "text-emerald-400 bg-emerald-400/10",
  SELL: "text-red-400 bg-red-400/10",
  HOLD: "text-blue-400 bg-blue-400/10",
};

// Tickers the user is watching for buy ideas. Each shows the advisory signal
// from ticker_signals (written daily by recommendation_job.py).
export default function Watchlist() {
  const { items, loading, error, addTicker, removeTicker, refresh } = useWatchlist();
  const { byTicker } = useTickerSignals();

  const [ticker, setTicker] = useState("");
  const [busy,   setBusy]   = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (!ticker) return;
    setBusy(true);
    const ok = await addTicker(ticker);
    setBusy(false);
    if (ok) setTicker("");
  }

  return (
    <div>
      <Header
        title="Watchlist"
        subtitle="Tickers you're tracking for buy ideas"
        onRefresh={refresh}
        loading={loading}
      />

      {error && (
        <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      <form onSubmit={submit} className="flex items-end gap-3 mb-6 bg-surface-card border border-surface-border rounded-xl p-4">
        <div>
          <label className="block text-xs text-slate-400 mb-1">Ticker</label>
          <input
            value={ticker}
            onChange={e => setTicker(e.target.value.toUpperCase())}
            placeholder="NVDA"
            className="w-32 bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-brand/50"
          />
        </div>
        <button
          type="submit"
          disabled={busy}
          className="flex items-center gap-2 bg-brand text-black font-semibold rounded-lg px-4 py-2 text-sm hover:brightness-110 transition-all disabled:opacity-50"
        >
          <Plus className="w-4 h-4" /> Add
        </button>
      </form>

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => <div key={i} className="h-28 bg-surface-card rounded-xl animate-pulse" />)}
        </div>
      ) : items.length ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {items.map(it => {
            const sig = byTicker[it.ticker?.toUpperCase()];
            return (
              <div key={it.id} className="bg-surface-card border border-surface-border rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-white text-lg">{it.ticker}</span>
                  <button onClick={() => removeTicker(it.id)} className="text-slate-600 hover:text-red-400 transition-colors">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                {sig ? (
                  <>
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`text-xs px-2 py-0.5 rounded-md font-medium ${ACTION_STYLE[sig.action]}`}>
                        {sig.action}
                      </span>
                      {sig.price != null && (
                        <span className="text-sm text-slate-300">${Number(sig.price).toFixed(2)}</span>
                      )}
                    </div>
                    {sig.rationale && <p className="text-xs text-slate-400">{sig.rationale}</p>}
                  </>
                ) : (
                  <p className="text-xs text-slate-600">Signal pending — updates after the daily job runs.</p>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-16 text-slate-500">
          <Eye className="w-6 h-6 mx-auto mb-2 opacity-40" />
          <p>Your watchlist is empty.</p>
          <p className="text-xs mt-1">Add a ticker above to track buy/sell signals.</p>
        </div>
      )}
    </div>
  );
}
