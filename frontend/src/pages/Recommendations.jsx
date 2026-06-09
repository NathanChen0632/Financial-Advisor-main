import { useRecommendations } from "../hooks/useRecommendations";
import Header from "../components/Layout/Header";
import { TrendingUp, Sparkles } from "lucide-react";

// Daily market-wide AI buy picks (from recommendation_job.py → recommendations table).
export default function Recommendations() {
  const { recs, batch, loading, error, refresh } = useRecommendations();

  const subtitle = batch
    ? `AI buy picks · batch ${batch}`
    : "AI buy picks — refreshed daily";

  return (
    <div>
      <Header
        title="Recommendations"
        subtitle={subtitle}
        onRefresh={refresh}
        loading={loading}
      />

      {error && (
        <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map(i => (
            <div key={i} className="h-44 bg-surface-card rounded-xl animate-pulse" />
          ))}
        </div>
      ) : recs.length ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {recs.map(r => <RecCard key={r.id} rec={r} />)}
        </div>
      ) : (
        <div className="text-center py-16 text-slate-500">
          <Sparkles className="w-6 h-6 mx-auto mb-2 opacity-40" />
          <p>No recommendations yet.</p>
          <p className="text-xs mt-1">
            The daily job (recommendation_job.py) writes AI buy picks here.
          </p>
        </div>
      )}
    </div>
  );
}

function RecCard({ rec }) {
  const metric = (label, val, suffix = "") =>
    val === null || val === undefined ? null : (
      <div className="flex justify-between text-xs">
        <span className="text-slate-500">{label}</span>
        <span className="text-slate-300">{val}{suffix}</span>
      </div>
    );

  return (
    <div className="bg-surface-card border border-surface-border rounded-xl p-5 hover:border-brand/40 transition-all">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-white text-lg">{rec.ticker}</span>
          <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-md bg-emerald-400/10 text-emerald-400 font-medium">
            <TrendingUp className="w-3 h-3" /> BUY
          </span>
        </div>
        {rec.price != null && (
          <span className="text-sm text-slate-300">${Number(rec.price).toFixed(2)}</span>
        )}
      </div>

      <div className="space-y-1 mb-3">
        {metric("Score",        rec.score != null ? Number(rec.score).toFixed(1) : null)}
        {metric("RSI (14)",     rec.rsi_14 != null ? Number(rec.rsi_14).toFixed(0) : null)}
        {metric("20d momentum", rec.mom_20d != null ? Number(rec.mom_20d).toFixed(1) : null, "%")}
        {metric("vs SPY",       rec.rel_strength != null ? Number(rec.rel_strength).toFixed(1) : null, "%")}
      </div>

      {rec.rationale && (
        <p className="text-xs text-slate-400 leading-relaxed border-t border-surface-border pt-3">
          {rec.rationale}
        </p>
      )}
    </div>
  );
}
