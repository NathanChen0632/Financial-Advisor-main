import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";
import { useAuth } from "../contexts/AuthContext";

// CRUD on the current user's `holdings` (paper positions). RLS scopes every
// row to auth.uid(); we still set user_id explicitly on insert.
export function useHoldings() {
  const { user } = useAuth();
  const [holdings, setHoldings] = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);

  useEffect(() => {
    if (!user) return;
    fetchHoldings();
    const channel = supabase
      .channel("holdings-feed")
      .on("postgres_changes", { event: "*", schema: "public", table: "holdings" }, fetchHoldings)
      .subscribe();
    return () => supabase.removeChannel(channel);
  }, [user?.id]);

  async function fetchHoldings() {
    setLoading(true);
    try {
      const { data, error: err } = await supabase
        .from("holdings")
        .select("*")
        .order("created_at", { ascending: false });
      if (err) throw err;
      setHoldings(data || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function addHolding({ ticker, qty, entry_price }) {
    setError(null);
    const { error: err } = await supabase.from("holdings").insert({
      user_id:     user.id,
      ticker:      ticker.toUpperCase().trim(),
      qty:         Number(qty),
      entry_price: Number(entry_price),
    });
    if (err) { setError(err.message); return false; }
    return true;
  }

  async function removeHolding(id) {
    const { error: err } = await supabase.from("holdings").delete().eq("id", id);
    if (err) setError(err.message);
  }

  return { holdings, loading, error, addHolding, removeHolding, refresh: fetchHoldings };
}
