# Bybit BTCUSDT 5m Scalping Bot (Python 3.12+)

Production-oriented intraday scalping bot for **Bybit** using:

- `ccxt` for exchange data/execution (REST + order routing)
- Bybit public WebSocket stream for low-latency live candle updates
- `pandas-ta-classic` indicators (RSI 10, MACD 8/21/5, BBands 20, ATR 14)
- `vectorbt` for vectorized backtesting, walk-forward analysis (WFO), and Monte Carlo
- Telegram alerts, Rich console output, Plotly charts

## 1) Setup (macOS M1 Sonoma compatible)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -V
pip install --upgrade pip
pip install -r requirements.txt
```

Expected Python version after activation: `3.12.x`.

Copy env template:

```bash
cp .env.example .env
```

Fill `.env` with your Bybit API key/secret and optional Telegram bot/chat keys.

API keys are required for all bot modes (`backtest` and `live`).

Only keys are stored in `.env`. Strategy/risk/filter settings are hardcoded in `vectorbt_bot.py` defaults.

### Auth troubleshooting (`retCode 10003`)

- `10003` usually means key/secret mismatch, expired key, or key restrictions (IP whitelist, account scope).
- Make sure key and secret are copied as a pair from the same API key.
- If your account/permissions are USDC-only, use a matching symbol (for example `BTC/USDC:USDC`) instead of `BTC/USDT:USDT`.

## 2) Backtest (last 5000 x 5m bars)

```bash
python vectorbt_bot.py backtest --bars 5000
```

With interactive Plotly chart:

```bash
python vectorbt_bot.py backtest --bars 5000 --plot
```

Backtest includes:

- slippage = `0.02%`
- fees default = taker `0.10%` per side (`DEFAULT_FEE_TAKER`), maker reference `0.04%`
- ATR-based SL/TP with `R:R = 1:2`
- risk sizing from `0.4%` risk per trade (default)
- WFO summary table
- Monte Carlo trade-bootstrap distribution

Optimization run:

```bash
python vectorbt_bot.py optimize --bars 5000 --top 10
```

Walk-forward out-of-sample optimization:

```bash
python vectorbt_bot.py optimize_wfo --bars 10000 --train 3000 --test 1000 --step 1000 --top 10
```

## 3) Live loop (5m, WebSocket-first)

Live mode places real orders by default:

```bash
python vectorbt_bot.py live --bars 500
```

Live engine details:

- Uses Bybit `kline` WebSocket topic and processes only **closed candles** (`confirm=true`)
- Keeps REST snapshot as warmup/fallback when stream is stale or reconnecting
- Auto reconnect with heartbeat for better stability
- Entry execution: **limit-first** (`post-only`) with timeout, then market fallback
- Market fallback has slippage/deviation guard (skip entry if price moved too far from signal)
- SL/TP are recalculated from actual fill price and submitted via Bybit position trading-stop
- Closed trades journal: writes only completed trades with non-zero realized PnL to `trade_journal.csv`

## 4) Strategy logic

- **Default profile (single mode):**
  - BUY: `RSI(10) < 35` + `MACD cross up` + `close > BB lower(20)`
  - SELL: `RSI(10) > 65` + `MACD cross down` + `close < BB upper(20)`
- **Stops/Targets:** ATR(14)-based SL, TP at 2x SL distance
- **Risk controls:**
  - `0.4%` risk per trade (default)
  - default max leverage `3.0x`
  - daily circuit breaker `+1.5% / -2%` (live loop)

## 5) Notes

- Runtime defaults (symbol, timeframe, risk, leverage) are defined in `vectorbt_bot.py` constants.
- If target metrics vary on a given sample (WR/PF), run multiple windows and compare WFO/MC robustness.
- This is educational software; validate behavior carefully before any production deployment.

## 6) File structure

- `requirements.txt`
- `vectorbt_bot.py`
- `.env.example`
- `README.md`
