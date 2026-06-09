# DQN Algorithmic Trading & Portfolio Optimizer
**CS5100 — Foundations of Artificial Intelligence**
Nathan Chen

A Deep Q-Network (DQN) reinforcement learning agent that learns when to buy and sell stocks using technical indicators and market data. Includes a live trading monitor, long straddle options strategy, Claude AI stock screener, portfolio optimizer, and a React dashboard.


## Project Structure

```
FINAL PROJECT/
├── backend/
│   ├── stock_prediction/
│   │   ├── main.py            — entry point, all CLI flags
│   │   ├── rl_agent.py        — DQN agent and trading environment
│   │   ├── features.py        — technical indicator feature engineering
│   │   ├── models.py          — data splitting and feature scaling
│   │   ├── data_collection.py — downloads stock data via yfinance
│   │   ├── backtesting.py     — equity curve, Sharpe ratio, drawdown metrics
│   │   ├── evaluation.py      — classification metrics and confusion matrix
│   │   ├── benchmarks.py      — rule-based strategies + DQN comparison
│   │   ├── walk_forward.py    — rolling window validation across time periods
│   │   ├── stress_test.py     — bear market stress testing (2008, 2020, 2022)
│   │   ├── portfolio.py       — portfolio screening and Markowitz optimization
│   │   ├── live_signal.py     — one-shot signal for today
│   │   ├── monitor.py         — continuous live monitoring loop
│   │   ├── alpaca_trader.py   — Alpaca API order execution
│   │   ├── straddle.py        — long straddle options strategy
│   │   ├── research_agent.py  — Claude AI stock screener
│   │   ├── supabase_bridge.py — writes signals/trades to Supabase
│   │   └── utils.py           — shared colours, .env loader, output dir
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── lib/supabase.js    — Supabase client
│   │   ├── hooks/             — usePortfolio, useSignals, usePerformance, useNews
│   │   ├── components/        — Layout, Portfolio, Recommendations, Performance, News
│   │   └── pages/             — Dashboard, Portfolio, Recommendations, Performance, News
│   ├── .env.local             — paste VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY here
│   └── package.json
├── database/
│   └── schema.sql             — run this in Supabase SQL Editor to create all tables
├── data/                      — cached CSV datasets
├── results/                   — output charts (auto-created on first run)
├── env/                       — Python virtual environment
├── .env                       — API credentials (never commit this)
└── final_paper.pdf
```


## Setup

**Backend:**
```bash
python3 -m venv env
source env/bin/activate
pip install -r backend/requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```


## Environment Variables

Create a `.env` file in the project root:

```
# Alpaca trading (paper by default)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true

# Claude AI stock research
ANTHROPIC_API_KEY=your_key_here

# Supabase — feeds the React dashboard
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here

# Email alerts (optional)
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
ALERT_TO=your_gmail@gmail.com
```

Create `frontend/.env.local`:
```
VITE_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key_here
```

**Alpaca keys:** alpaca.markets → Paper Trading → API Keys
**Gmail app password:** myaccount.google.com/security → App passwords


## Backend Commands

All commands run from the project root.

**Backtest (default tickers AAPL, MSFT, TSLA):**
```bash
python3 backend/stock_prediction/main.py
```

**Backtest a specific ticker:**
```bash
python3 backend/stock_prediction/main.py --ticker NVDA
```

**Live signal for today:**
```bash
python3 backend/stock_prediction/main.py --signal --ticker AAPL
```

**Paper trade — DQN signals, no real money:**
```bash
python3 backend/stock_prediction/main.py --monitor --alpaca
```

**Paper trade with email alerts:**
```bash
python3 backend/stock_prediction/main.py --monitor --alpaca --email
```

**Add long straddle options strategy:**
```bash
python3 backend/stock_prediction/main.py --monitor --alpaca --straddle
```

**Full stack — Claude screens stocks, DQN trades equities, straddle catches vol events:**
```bash
python3 backend/stock_prediction/main.py --research --monitor --alpaca --straddle
```

**Live trade (real money — be careful):**
```bash
python3 backend/stock_prediction/main.py --monitor --alpaca --live
```

**Portfolio optimizer:**
```bash
python3 backend/stock_prediction/main.py --portfolio
```

**Strategy benchmark comparison (DQN vs MA crossover, RSI, momentum):**
```bash
python3 backend/stock_prediction/main.py --benchmark
```

**Walk-forward validation across 3 time periods:**
```bash
python3 backend/stock_prediction/main.py --walk-forward
```

**Bear market stress tests (2008, 2020, 2022):**
```bash
python3 backend/stock_prediction/main.py --stress-test
```

**Broad 30+ ticker universe:**
```bash
python3 backend/stock_prediction/main.py --broad --benchmark --walk-forward
```

Results (equity curves, confusion matrices) are saved to `results/`.


## Frontend (React Dashboard)

1. Run the schema SQL in Supabase → SQL Editor → New Query (paste `database/schema.sql`)
2. Fill in `frontend/.env.local` with your Supabase URL and anon key
3. Start the dashboard:
```bash
cd frontend && npm run dev
```

The dashboard reads from Supabase in real time. The Python monitor writes signals, positions, and trades there automatically via `supabase_bridge.py`.

| Page | What it shows |
|---|---|
| Dashboard | Live KPIs, equity curve, active BUY/SELL signals |
| Portfolio | Open positions + completed trade history |
| Recommendations | Full DQN signal feed with action filters |
| Performance | Equity curve vs buy-and-hold, Sharpe, drawdown |
| News | Headlines from the Claude research agent |


## How It Works

The DQN agent trains on 10 years of daily price data (2015–2025). It observes 31 features per day — RSI, MACD, Bollinger Bands, moving averages, ATR, volume, and SPY market context — and learns when to buy or move to cash.

The trading environment enforces real rules: a minimum 2:1 reward-to-risk ratio, ATR-based stop losses, volume confirmation, and a 10-day maximum hold. This teaches the agent discipline rather than just pattern matching.

The long straddle strategy runs alongside the DQN. It buys an ATM call + put when IV is low and a big move is expected, then exits at a 40% profit target or before theta decay accelerates.

The portfolio builder trains a separate DQN per stock, selects the top performers by Sharpe ratio, then uses Markowitz mean-variance optimization to find the capital allocation that maximizes the combined Sharpe ratio.
