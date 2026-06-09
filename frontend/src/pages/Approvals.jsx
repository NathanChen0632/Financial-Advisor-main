import Header from "../components/Layout/Header";
import { useApprovals } from "../hooks/useApprovals";
import { useAuth } from "../contexts/AuthContext";
import { Check, X, ShieldCheck, Lock } from "lucide-react";

const STATUS_STYLE = {
  approved: "text-emerald-400 bg-emerald-400/10",
  rejected: "text-slate-400 bg-slate-400/10",
  executed: "text-blue-400 bg-blue-400/10",
};

// Master-only. The algorithm requests approval before selling; the master
// approves or rejects here, and the monitor acts on the decision.
export default function Approvals() {
  const { isMaster } = useAuth();
  if (!isMaster) {
    return (
      <div className="text-center py-20 text-slate-500">
        <Lock className="w-7 h-7 mx-auto mb-3 opacity-40" />
        <p className="text-white font-medium">Approvals are master-only</p>
        <p className="text-xs mt-1">Only the trading account approves sells.</p>
      </div>
    );
  }
  return <ApprovalsInner />;
}

function ApprovalsInner() {
  const { pending, history, loading, error, approve, reject, refresh } = useApprovals();

  return (
    <div>
      <Header title="Sell Approvals" subtitle="The algorithm needs your OK before it sells" onRefresh={refresh} loading={loading} />

      {error && (
        <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-4 py-3">{error}</div>
      )}

      {/* Pending */}
      {pending.length ? (
        <div className="space-y-3 mb-8">
          {pending.map(a => (
            <div key={a.id} className="bg-surface-card border border-yellow-400/30 rounded-xl p-4 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white text-lg">{a.ticker}</span>
                  <span className="text-xs px-2 py-0.5 rounded-md bg-red-400/10 text-red-400 font-medium">SELL</span>
                  {a.suggested_price != null && (
                    <span className="text-sm text-slate-400">~${Number(a.suggested_price).toFixed(2)}</span>
                  )}
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  {a.reason || "strategy signal"} · {new Date(a.created_at).toLocaleString()}
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => approve(a.id)}
                  className="flex items-center gap-1.5 bg-emerald-500 text-black font-semibold rounded-lg px-3 py-2 text-sm hover:brightness-110 transition-all"
                >
                  <Check className="w-4 h-4" /> Approve
                </button>
                <button
                  onClick={() => reject(a.id)}
                  className="flex items-center gap-1.5 bg-surface border border-surface-border text-slate-300 rounded-lg px-3 py-2 text-sm hover:text-white hover:border-slate-600 transition-all"
                >
                  <X className="w-4 h-4" /> Reject
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-slate-500 mb-8">
          <ShieldCheck className="w-6 h-6 mx-auto mb-2 opacity-40" />
          <p>No sells awaiting approval.</p>
          <p className="text-xs mt-1">When the algorithm wants to sell, it'll appear here and email you.</p>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <>
          <h2 className="text-sm font-semibold text-white mb-2">Recent decisions</h2>
          <div className="bg-surface-card border border-surface-border rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead className="text-xs text-slate-500 border-b border-surface-border">
                <tr>
                  <th className="text-left px-4 py-2.5">Ticker</th>
                  <th className="text-left px-4 py-2.5">Reason</th>
                  <th className="text-left px-4 py-2.5">Status</th>
                  <th className="text-left px-4 py-2.5">Decided</th>
                </tr>
              </thead>
              <tbody>
                {history.map(a => (
                  <tr key={a.id} className="border-b border-surface-border/50 last:border-0">
                    <td className="px-4 py-2.5 text-white font-medium">{a.ticker}</td>
                    <td className="px-4 py-2.5 text-slate-400 text-xs">{a.reason}</td>
                    <td className="px-4 py-2.5">
                      <span className={`text-xs px-2 py-0.5 rounded-md font-medium ${STATUS_STYLE[a.status] || ""}`}>{a.status}</span>
                    </td>
                    <td className="px-4 py-2.5 text-slate-500 text-xs">{a.decided_at ? new Date(a.decided_at).toLocaleString() : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
