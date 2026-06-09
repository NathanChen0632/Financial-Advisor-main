import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";
import { useAuth } from "../contexts/AuthContext";

// CRUD on the current user's `watchlist`. RLS scopes rows to auth.uid().
export function useWatchlist() {
  const { user } = useAuth();
  const [items,   setItems]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    if (!user) return;
    fetchItems();
    const channel = supabase
      .channel("watchlist-feed")
      .on("postgres_changes", { event: "*", schema: "public", table: "watchlist" }, fetchItems)
      .subscribe();
    return () => supabase.removeChannel(channel);
  }, [user?.id]);

  async function fetchItems() {
    setLoading(true);
    try {
      const { data, error: err } = await supabase
        .from("watchlist")
        .select("*")
        .order("created_at", { ascending: false });
      if (err) throw err;
      setItems(data || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function addTicker(ticker) {
    setError(null);
    const { error: err } = await supabase.from("watchlist").insert({
      user_id: user.id,
      ticker:  ticker.toUpperCase().trim(),
    });
    if (err) { setError(err.message); return false; }
    return true;
  }

  async function removeTicker(id) {
    const { error: err } = await supabase.from("watchlist").delete().eq("id", id);
    if (err) setError(err.message);
  }

  return { items, loading, error, addTicker, removeTicker, refresh: fetchItems };
}
