import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";

// Reads from the `positions` and `trades` tables.
// The Python backend writes here via supabase_bridge.py whenever
// a position is opened, updated, or closed.
export function usePortfolio() {
  const [positions, setPositions] = useState([]);
  const [trades,    setTrades]    = useState([]);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);

  useEffect(() => {
    fetchAll();

    // Subscribe to real-time changes so the UI updates when the Python
    // monitor writes a new position or closes one.
    const channel = supabase
      .channel("portfolio-changes")
      .on("postgres_changes", { event: "*", schema: "public", table: "positions" }, fetchAll)
      .on("postgres_changes", { event: "*", schema: "public", table: "trades" },    fetchAll)
      .subscribe();

    return () => supabase.removeChannel(channel);
  }, []);

  async function fetchAll() {
    setLoading(true);
    try {
      const [posRes, tradeRes] = await Promise.all([
        supabase.from("positions").select("*").order("entry_time", { ascending: false }),
        supabase.from("trades").select("*").order("exit_time",  { ascending: false }).limit(50),
      ]);
      if (posRes.error)   throw posRes.error;
      if (tradeRes.error) throw tradeRes.error;
      setPositions(posRes.data   || []);
      setTrades(tradeRes.data    || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const totalPnl    = positions.reduce((s, p) => s + (p.unrealized_pnl || 0), 0);
  const totalValue  = positions.reduce((s, p) => s + (p.market_value   || 0), 0);
  const winRate     = trades.length
    ? (trades.filter(t => t.pnl > 0).length / trades.length) * 100
    : 0;

  return { positions, trades, loading, error, totalPnl, totalValue, winRate, refresh: fetchAll };
}
