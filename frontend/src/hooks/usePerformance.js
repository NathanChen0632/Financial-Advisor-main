import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";

// Reads from the `performance_snapshots` table.
// The Python bridge writes a snapshot after each trading session with
// daily equity, returns, Sharpe ratio, and drawdown.
export function usePerformance(days = 90) {
  const [snapshots, setSnapshots] = useState([]);
  const [summary,   setSummary]   = useState(null);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);

  useEffect(() => { fetchPerformance(); }, [days]);

  async function fetchPerformance() {
    setLoading(true);
    try {
      const since = new Date();
      since.setDate(since.getDate() - days);

      const { data, error: err } = await supabase
        .from("performance_snapshots")
        .select("*")
        .gte("snapshot_date", since.toISOString().slice(0, 10))
        .order("snapshot_date", { ascending: true });

      if (err) throw err;
      const rows = data || [];
      setSnapshots(rows);

      if (rows.length >= 2) {
        const first = rows[0];
        const last  = rows[rows.length - 1];
        const totalReturn  = ((last.equity - first.equity) / first.equity) * 100;
        const dailyReturns = rows.slice(1).map((r, i) =>
          (r.equity - rows[i].equity) / rows[i].equity
        );
        const mean = dailyReturns.reduce((s, r) => s + r, 0) / dailyReturns.length;
        const std  = Math.sqrt(dailyReturns.reduce((s, r) => s + (r - mean) ** 2, 0) / dailyReturns.length);
        const sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0;

        let peak = -Infinity, maxDD = 0;
        rows.forEach(r => {
          if (r.equity > peak) peak = r.equity;
          const dd = (r.equity - peak) / peak;
          if (dd < maxDD) maxDD = dd;
        });

        setSummary({
          currentEquity: last.equity,
          totalReturn:   totalReturn.toFixed(2),
          sharpe:        sharpe.toFixed(3),
          maxDrawdown:   (maxDD * 100).toFixed(2),
          startEquity:   first.equity,
        });
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return { snapshots, summary, loading, error, refresh: fetchPerformance };
}
