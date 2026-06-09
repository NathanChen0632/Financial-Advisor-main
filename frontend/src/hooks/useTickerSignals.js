import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";

// Reads the `ticker_signals` table (advisory HOLD/SELL/BUY per ticker, written
// by the daily recommendation_job.py) and returns a { TICKER: row } map so
// holdings/watchlist pages can look up the suggestion for each symbol.
export function useTickerSignals() {
  const [byTicker, setByTicker] = useState({});
  const [loading,  setLoading]  = useState(true);

  useEffect(() => {
    fetchSignals();
    const channel = supabase
      .channel("ticker-signals-feed")
      .on("postgres_changes", { event: "*", schema: "public", table: "ticker_signals" }, fetchSignals)
      .subscribe();
    return () => supabase.removeChannel(channel);
  }, []);

  async function fetchSignals() {
    setLoading(true);
    try {
      const { data } = await supabase.from("ticker_signals").select("*");
      const map = {};
      for (const row of data || []) map[row.ticker.toUpperCase()] = row;
      setByTicker(map);
    } finally {
      setLoading(false);
    }
  }

  return { byTicker, loading, refresh: fetchSignals };
}
