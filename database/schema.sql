-- ============================================================
-- DQN Trading Dashboard — Supabase Schema
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================


-- 1. SIGNALS
--    Written by the Python monitor every poll cycle.
--    One row per (ticker, poll_time) — keeps the full history.
-- ------------------------------------------------------------
create table if not exists signals (
  id           bigint  generated always as identity primary key,
  ticker       text    not null,
  action       text    not null check (action in ('BUY','SELL','HOLD','WAIT')),
  price        numeric(12,4),
  stop_price   numeric(12,4),
  target_price numeric(12,4),
  atr_pct      numeric(8,6),
  pnl_pct      numeric(8,4),      -- only populated on SELL
  exit_reason  text,               -- 'stop hit' | 'target hit' | 'time stop' | 'signal'
  strategy     text default 'DQN',
  created_at   timestamptz default now()
);

create index if not exists signals_ticker_idx    on signals (ticker);
create index if not exists signals_created_at_idx on signals (created_at desc);

-- Enable real-time so the React app auto-updates
alter publication supabase_realtime add table signals;


-- 2. POSITIONS
--    One row per open position. Updated while held, deleted on close.
-- ------------------------------------------------------------
create table if not exists positions (
  id               bigint  generated always as identity primary key,
  ticker           text    not null unique,
  strategy_type    text    default 'DQN Equity',   -- or 'Straddle'
  qty              integer,
  entry_price      numeric(12,4),
  current_price    numeric(12,4),
  stop_price       numeric(12,4),
  target_price     numeric(12,4),
  market_value     numeric(14,2),
  unrealized_pnl   numeric(12,2),
  unrealized_pnl_pct numeric(8,4),
  entry_time       timestamptz default now(),
  updated_at       timestamptz default now()
);

alter publication supabase_realtime add table positions;


-- 3. TRADES
--    Completed trade history. Written when a position is closed.
-- ------------------------------------------------------------
create table if not exists trades (
  id           bigint  generated always as identity primary key,
  ticker       text    not null,
  strategy_type text   default 'DQN Equity',
  qty          integer,
  entry_price  numeric(12,4),
  exit_price   numeric(12,4),
  pnl          numeric(12,2),
  pnl_pct      numeric(8,4),
  exit_reason  text,
  entry_time   timestamptz,
  exit_time    timestamptz default now()
);

create index if not exists trades_ticker_idx   on trades (ticker);
create index if not exists trades_exit_time_idx on trades (exit_time desc);

alter publication supabase_realtime add table trades;


-- 4. PERFORMANCE SNAPSHOTS
--    Written once per day (or per session) by the Python bridge.
--    Powers the equity curve and Sharpe/drawdown metrics.
-- ------------------------------------------------------------
create table if not exists performance_snapshots (
  id            bigint  generated always as identity primary key,
  snapshot_date date    not null unique,
  equity        numeric(14,2) not null,
  bah_equity    numeric(14,2),    -- buy-and-hold baseline on same capital
  daily_return  numeric(8,6),
  sharpe_ytd    numeric(8,4),
  max_drawdown  numeric(8,4),
  created_at    timestamptz default now()
);

create index if not exists perf_date_idx on performance_snapshots (snapshot_date);


-- 5. NEWS ITEMS
--    Written by research_agent.py when it screens stocks.
--    headline/summary come from the Claude API response.
-- ------------------------------------------------------------
create table if not exists news_items (
  id           bigint  generated always as identity primary key,
  ticker       text,               -- null = market-wide news
  headline     text    not null,
  summary      text,
  source       text,
  url          text,
  sentiment    text check (sentiment in ('positive','negative','neutral')),
  published_at timestamptz default now(),
  created_at   timestamptz default now()
);

create index if not exists news_ticker_idx      on news_items (ticker);
create index if not exists news_published_at_idx on news_items (published_at desc);


-- ============================================================
-- Row-Level Security (RLS)
-- The anon key used in the frontend can only READ data.
-- The Python backend uses the service-role key to WRITE.
-- ============================================================

alter table signals               enable row level security;
alter table positions             enable row level security;
alter table trades                enable row level security;
alter table performance_snapshots enable row level security;
alter table news_items            enable row level security;

-- Public read-only for the React app
create policy "public read signals"    on signals               for select using (true);
create policy "public read positions"  on positions             for select using (true);
create policy "public read trades"     on trades                for select using (true);
create policy "public read perf"       on performance_snapshots for select using (true);
create policy "public read news"       on news_items            for select using (true);

-- ============================================================
-- Done. Next step: add SUPABASE_SERVICE_KEY to your .env and
-- run supabase_bridge.py alongside the Python monitor.
-- ============================================================
