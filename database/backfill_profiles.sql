-- ============================================================
-- One-time backfill: create a profile row for every EXISTING auth user.
-- The signup trigger only fires for new users, so accounts created before
-- the profiles table existed (e.g. the master) have no row yet — which makes
-- the master-only RLS policies reject their writes.
--
-- Run once in: Supabase → SQL Editor → New query → paste → Run.
-- Safe to re-run.
-- ============================================================
insert into public.profiles (id, email, role)
select
  id,
  email,
  case when lower(email) = lower('nathanchen32@gmail.com') then 'master' else 'user' end
from auth.users
on conflict (id) do update
  set role  = excluded.role,
      email = excluded.email;

-- Show the result so you can confirm the master role stuck.
select email, role from public.profiles order by role;
