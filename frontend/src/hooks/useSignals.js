import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";

// Reads from the `signals` table.
// The Python monitor writes a row here every time it polls and produces
// a BUY, SELL, HOLD, or WAIT signal for a ticker.
export function useSignals(limit = 30) {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetchSignals();

    const channel = supabase
      .channel("signal-feed")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "signals" }, fetchSignals)
      .subscribe();

    return () => supabase.removeChannel(channel);
  }, [limit]);

  async function fetchSignals() {
    setLoading(true);
    try {
      const { data, error: err } = await supabase
        .from("signals")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(limit);
      if (err) throw err;
      setSignals(data || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  // Deduplicate to show latest signal per ticker
  const latestByTicker = Object.values(
    signals.reduce((acc, s) => {
      if (!acc[s.ticker] || s.created_at > acc[s.ticker].created_at) acc[s.ticker] = s;
      return acc;
    }, {})
  );

  return { signals, latestByTicker, loading, error, refresh: fetchSignals };
}
