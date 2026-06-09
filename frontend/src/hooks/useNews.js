import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";

// Reads from the `news_items` table.
// The Python research_agent.py writes market news here when it screens stocks.
// Alternatively you can backfill this table from any news API.
export function useNews(ticker = null, limit = 20) {
  const [news,    setNews]    = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => { fetchNews(); }, [ticker, limit]);

  async function fetchNews() {
    setLoading(true);
    try {
      let query = supabase
        .from("news_items")
        .select("*")
        .order("published_at", { ascending: false })
        .limit(limit);

      if (ticker) query = query.eq("ticker", ticker);

      const { data, error: err } = await query;
      if (err) throw err;
      setNews(data || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return { news, loading, error, refresh: fetchNews };
}
