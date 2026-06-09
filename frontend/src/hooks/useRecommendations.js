import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";

// Reads the `recommendations` table — daily market-wide AI buy picks written
// by the backend recommendation_job.py. Shows the most recent batch.
export function useRecommendations() {
  const [recs,    setRecs]    = useState([]);
  const [batch,   setBatch]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetchRecs();

    const channel = supabase
      .channel("recommendations-feed")
      .on("postgres_changes", { event: "*", schema: "public", table: "recommendations" }, fetchRecs)
      .subscribe();

    return () => supabase.removeChannel(channel);
  }, []);

  async function fetchRecs() {
    setLoading(true);
    try {
      // Pull recent rows ordered by batch then score, then keep only the
      // latest batch_date so the page always shows one coherent set of picks.
      const { data, error: err } = await supabase
        .from("recommendations")
        .select("*")
        .order("batch_date", { ascending: false })
        .order("score",      { ascending: false })
        .limit(100);
      if (err) throw err;

      const rows  = data || [];
      const newest = rows.length ? rows[0].batch_date : null;
      setBatch(newest);
      setRecs(rows.filter(r => r.batch_date === newest));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return { recs, batch, loading, error, refresh: fetchRecs };
}
