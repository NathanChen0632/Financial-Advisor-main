import { useState } from "react";
import { Activity, Loader2 } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";

// Email/password sign in + sign up via Supabase Auth.
// Shown whenever there is no active session (see App.jsx gating).
export default function Login() {
  const { signIn, signUp } = useAuth();
  const [mode,    setMode]    = useState("signin"); // 'signin' | 'signup'
  const [email,   setEmail]   = useState("");
  const [password,setPassword]= useState("");
  const [busy,    setBusy]    = useState(false);
  const [error,   setError]   = useState(null);
  const [notice,  setNotice]  = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      const fn = mode === "signin" ? signIn : signUp;
      const { data, error: err } = await fn(email.trim(), password);
      if (err) throw err;
      // On the onAuthStateChange listener picks up the new session and the
      // app re-renders into the dashboard. If email confirmation is on,
      // signUp returns no session — tell the user to confirm.
      if (mode === "signup" && !data.session) {
        setNotice("Account created. Check your email to confirm, then sign in.");
        setMode("signin");
      }
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface px-4">
      <div className="w-full max-w-sm">
        <div className="flex items-center gap-2 justify-center mb-6">
          <Activity className="w-6 h-6 text-brand" />
          <span className="font-semibold text-white text-lg tracking-wide">DQN Trader</span>
        </div>

        <div className="bg-surface-card border border-surface-border rounded-xl p-6">
          <h1 className="text-lg font-semibold text-white mb-1">
            {mode === "signin" ? "Sign in" : "Create account"}
          </h1>
          <p className="text-sm text-slate-500 mb-5">
            {mode === "signin"
              ? "Welcome back. Sign in to your dashboard."
              : "Sign up to get AI stock suggestions."}
          </p>

          {error && (
            <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-3 py-2">
              {error}
            </div>
          )}
          {notice && (
            <div className="mb-4 bg-brand-dim border border-brand/30 text-brand text-sm rounded-lg px-3 py-2">
              {notice}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-3">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Email</label>
              <input
                type="email"
                required
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-brand/50"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Password</label>
              <input
                type="password"
                required
                minLength={6}
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-brand/50"
              />
            </div>

            <button
              type="submit"
              disabled={busy}
              className="w-full flex items-center justify-center gap-2 bg-brand text-black font-semibold rounded-lg px-3 py-2.5 text-sm hover:brightness-110 transition-all disabled:opacity-50"
            >
              {busy && <Loader2 className="w-4 h-4 animate-spin" />}
              {mode === "signin" ? "Sign in" : "Sign up"}
            </button>
          </form>

          <button
            onClick={() => { setMode(mode === "signin" ? "signup" : "signin"); setError(null); setNotice(null); }}
            className="w-full text-center text-xs text-slate-500 hover:text-slate-300 mt-4"
          >
            {mode === "signin"
              ? "Don't have an account? Sign up"
              : "Already have an account? Sign in"}
          </button>
        </div>
      </div>
    </div>
  );
}
