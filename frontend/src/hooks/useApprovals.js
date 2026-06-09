import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";

// Master-only. Lists pending sell approvals (written by the monitor when the
// algorithm wants to sell) and lets the master approve/reject. The monitor
// executes approved rows on its next tick.
export function useApprovals() {
  const [pending, setPending] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetchApprovals();
    const channel = supabase
      .channel("approvals-feed")
      .on("postgres_changes", { event: "*", schema: "public", table: "sell_approvals" }, fetchApprovals)
      .subscribe();
    return () => supabase.removeChannel(channel);
  }, []);

  async function fetchApprovals() {
    setLoading(true);
    try {
      const { data, error: err } = await supabase
        .from("sell_approvals")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(50);
      if (err) throw err;
      const rows = data || [];
      setPending(rows.filter(r => r.status === "pending"));
      setHistory(rows.filter(r => r.status !== "pending"));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function decide(id, status) {
    setError(null);
    const { error: err } = await supabase
      .from("sell_approvals")
      .update({ status, decided_at: new Date().toISOString() })
      .eq("id", id);
    if (err) setError(err.message);
  }

  return {
    pending, history, loading, error,
    approve: id => decide(id, "approved"),
    reject:  id => decide(id, "rejected"),
    refresh: fetchApprovals,
  };
}
