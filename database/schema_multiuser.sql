-- ============================================================
-- DQN Trading Dashboard — Multi-user extension
-- Run this in Supabase: SQL Editor → New query → paste → Run.
--
-- You do NOT need to re-run schema.sql — its tables already exist.
-- This script is idempotent: safe to run more than once.
--
-- Adds auth-aware tables (profiles, holdings, watchlist, orders,
-- approvals, recommendations, ticker_signals), their RLS policies,
-- and a trigger that creates a profile + role on signup.
-- ============================================================


-- 1. PROFILES — one row per auth user, carries the role
-- ------------------------------------------------------------
create table if not exists profiles (
  id         uuid primary key references auth.users (id) on delete cascade,
  email      text,
  role       text not null default 'user' check (role in ('master','user')),
  created_at timestamptz default now()
);

-- Auto-create a profile when a user signs up. The master email gets
-- the 'master' role; everyone else is a suggestion-only 'user'.
create or replace function handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, email, role)
  values (
    new.id,
    new.email,
    case when lower(new.email) = lower('nathanchen32@gmail.com') then 'master' else 'user' end
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function handle_new_user();


-- 2. ORDER_REQUESTS — UI-initiated trades (master only).
-- ------------------------------------------------------------
create table if not exists order_requests (
  id           bigint generated always as identity primary key,
  user_id      uuid references auth.users (id) on delete set null,
  ticker       text not null,
  side         text not null check (side in ('buy','sell')),
  qty          integer,
  status       text not null default 'pending'
               check (status in ('pending','executing','filled','rejected','error')),
  error        text,
  created_at   timestamptz default now(),
  processed_at timestamptz
);
create index if not exists order_requests_status_idx on order_requests (status);


-- 3. SELL_APPROVALS — algorithmic sells awaiting master approval.
-- ------------------------------------------------------------
create table if not exists sell_approvals (
  id              bigint generated always as identity primary key,
  ticker          text not null,
  reason          text,
  suggested_price numeric(12,4),
  status          text not null default 'pending'
                  check (status in ('pending','approved','rejected','executed')),
  created_at      timestamptz default now(),
  decided_at      timestamptz
);
create index if not exists sell_approvals_status_idx on sell_approvals (status);


-- 4. HOLDINGS — per-user paper positions (suggestion-only users)
-- ------------------------------------------------------------
create table if not exists holdings (
  id          bigint generated always as identity primary key,
  user_id     uuid not null references auth.users (id) on delete cascade,
  ticker      text not null,
  qty         numeric(14,4),
  entry_price numeric(12,4),
  entry_time  timestamptz default now(),
  created_at  timestamptz default now()
);
create index if not exists holdings_user_idx on holdings (user_id);


-- 5. WATCHLIST — per-user tickers to watch for buy ideas
-- ------------------------------------------------------------
create table if not exists watchlist (
  id         bigint generated always as identity primary key,
  user_id    uuid not null references auth.users (id) on delete cascade,
  ticker     text not null,
  created_at timestamptz default now(),
  unique (user_id, ticker)
);
create index if not exists watchlist_user_idx on watchlist (user_id);


-- 6. RECOMMENDATIONS — daily market-wide AI buy picks (Claude)
-- ------------------------------------------------------------
create table if not exists recommendations (
  id           bigint generated always as identity primary key,
  ticker       text not null,
  rationale    text,
  score        numeric(10,2),
  price        numeric(12,4),
  rsi_14       numeric(8,2),
  mom_20d      numeric(8,2),
  rel_strength numeric(8,2),
  batch_date   date not null default current_date,
  created_at   timestamptz default now(),
  unique (ticker, batch_date)
);
create index if not exists recommendations_batch_idx on recommendations (batch_date desc);


-- 7. TICKER_SIGNALS — advisory HOLD/SELL/BUY per ticker (rule-based)
-- ------------------------------------------------------------
create table if not exists ticker_signals (
  id         bigint generated always as identity primary key,
  ticker     text not null unique,
  action     text not null check (action in ('BUY','SELL','HOLD')),
  rationale  text,
  rsi        numeric(8,2),
  momentum   numeric(8,2),
  price      numeric(12,4),
  updated_at timestamptz default now()
);


-- ------------------------------------------------------------
-- Real-time: add the new tables to the supabase_realtime publication.
-- Wrapped so re-running doesn't error if a table is already a member.
-- ------------------------------------------------------------
do $$
declare t text;
begin
  foreach t in array array[
    'order_requests','sell_approvals','holdings',
    'watchlist','recommendations','ticker_signals'
  ]
  loop
    begin
      execute format('alter publication supabase_realtime add table %I', t);
    exception when duplicate_object then
      null;  -- already a member; ignore
    end;
  end loop;
end $$;


-- ============================================================
-- Row-Level Security
-- Frontend uses the anon key + a logged-in session (auth.uid()).
-- Backend uses the service-role key, which bypasses RLS entirely.
-- ============================================================
alter table profiles        enable row level security;
alter table order_requests  enable row level security;
alter table sell_approvals  enable row level security;
alter table holdings        enable row level security;
alter table watchlist       enable row level security;
alter table recommendations enable row level security;
alter table ticker_signals  enable row level security;

-- Drop-then-create so re-running doesn't error on existing policies.
drop policy if exists "read own profile"   on profiles;
drop policy if exists "update own profile" on profiles;
drop policy if exists "own holdings"       on holdings;
drop policy if exists "own watchlist"      on watchlist;
drop policy if exists "master orders"      on order_requests;
drop policy if exists "master approvals"   on sell_approvals;
drop policy if exists "auth read recs"     on recommendations;
drop policy if exists "auth read signals"  on ticker_signals;

-- profiles: a user can read (and update) only their own row
create policy "read own profile"   on profiles for select using (auth.uid() = id);
create policy "update own profile" on profiles for update using (auth.uid() = id);

-- holdings / watchlist: full CRUD, but only on your own rows
create policy "own holdings"  on holdings  for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own watchlist" on watchlist for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- order_requests / sell_approvals: master only
create policy "master orders" on order_requests for all
  using     (exists (select 1 from profiles p where p.id = auth.uid() and p.role = 'master'))
  with check (exists (select 1 from profiles p where p.id = auth.uid() and p.role = 'master'));
create policy "master approvals" on sell_approvals for all
  using     (exists (select 1 from profiles p where p.id = auth.uid() and p.role = 'master'))
  with check (exists (select 1 from profiles p where p.id = auth.uid() and p.role = 'master'));

-- recommendations / ticker_signals: any signed-in user may read
create policy "auth read recs"    on recommendations for select using (auth.uid() is not null);
create policy "auth read signals" on ticker_signals  for select using (auth.uid() is not null);

-- ============================================================
-- Done. Backend writes (service-role key) bypass RLS.
-- ============================================================
