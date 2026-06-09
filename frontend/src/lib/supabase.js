import { createClient } from "@supabase/supabase-js";

// Paste your Project URL and anon public key from:
// Supabase Dashboard → Settings → API
const SUPABASE_URL    = import.meta.env.VITE_SUPABASE_URL    || "https://YOUR_PROJECT.supabase.co";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || "YOUR_ANON_KEY";

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
