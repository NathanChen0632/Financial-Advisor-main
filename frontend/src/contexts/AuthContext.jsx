import { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "../lib/supabase";

// The account that trades for real on Alpaca. Everyone else is
// suggestion-only. Kept in sync with the signup trigger in
// database/schema_multiuser.sql.
export const MASTER_EMAIL = "nathanchen32@gmail.com";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get the current session on mount, then subscribe to changes.
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setLoading(false);
    });

    const { data: sub } = supabase.auth.onAuthStateChange((_event, sess) => {
      setSession(sess);
    });

    return () => sub.subscription.unsubscribe();
  }, []);

  // Whenever the session changes, load the matching profile row (for the role).
  useEffect(() => {
    if (!session?.user) {
      setProfile(null);
      return;
    }
    let cancelled = false;
    (async () => {
      const { data } = await supabase
        .from("profiles")
        .select("*")
        .eq("id", session.user.id)
        .maybeSingle();
      if (!cancelled) setProfile(data);
    })();
    return () => { cancelled = true; };
  }, [session]);

  // Role resolution: trust the profile row, but fall back to the email
  // match so the UI works even before the profile row has loaded.
  const email  = session?.user?.email ?? null;
  const role   = profile?.role
    ?? (email && email.toLowerCase() === MASTER_EMAIL.toLowerCase() ? "master" : email ? "user" : null);
  const isMaster = role === "master";

  const value = {
    session,
    user: session?.user ?? null,
    profile,
    role,
    isMaster,
    loading,
    signUp: (e, p) => supabase.auth.signUp({ email: e, password: p }),
    signIn: (e, p) => supabase.auth.signInWithPassword({ email: e, password: p }),
    signOut: () => supabase.auth.signOut(),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider>");
  return ctx;
}
