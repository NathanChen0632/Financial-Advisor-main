import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";
import { useAuth } from "../contexts/AuthContext";

// Master-only. Submits trades to the `order_requests` queue, which the running
// monitor executes on Alpaca. RLS already restricts this table to the master.
export function useOrderRequests() {
  const { user } = useAuth();
  const [orders,  setOrders]  = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetchOrders();
    const channel = supabase
      .channel("order-requests-feed")
      .on("postgres_changes", { event: "*", schema: "public", table: "order_requests" }, fetchOrders)
      .subscribe();
    return () => supabase.removeChannel(channel);
  }, []);

  async function fetchOrders() {
    setLoading(true);
    try {
      const { data, error: err } = await supabase
        .from("order_requests")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(50);
      if (err) throw err;
      setOrders(data || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitOrder({ ticker, side, qty }) {
    setError(null);
    const row = {
      user_id: user.id,
      ticker:  ticker.toUpperCase().trim(),
      side,
      qty: qty ? Number(qty) : null,
    };
    const { error: err } = await supabase.from("order_requests").insert(row);
    if (err) { setError(err.message); return false; }
    return true;
  }

  return { orders, loading, error, submitOrder, refresh: fetchOrders };
}
