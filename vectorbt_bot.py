#!/usr/bin/env python3
"""
Bybit BTCUSDT 5m intraday scalping bot (Python 3.12+).

Features:
- CCXT Bybit integration
- Indicators: RSI(10), MACD(8,21,5), BBands(20), ATR(14)
- VectorBT backtest (slippage, WFO, Monte Carlo)
- ATR-based SL/TP with 1:2 R:R
- Risk model: 0.4% risk per trade + daily +1.5% / -2% circuit breaker (live)
- Telegram alerts
- Rich console output + Plotly charts
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import signal
import sys
import threading
import time
import json
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import requests
import vectorbt as vbt
import websocket
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table
import pandas_ta_classic as ta


# Runtime defaults (kept in code, not in .env)
DEFAULT_SYMBOL = "BTC/USDT:USDT"
DEFAULT_TIMEFRAME = "5m"
DEFAULT_RISK_PER_TRADE = 0.004
DEFAULT_DAILY_PROFIT_STOP = 0.015
DEFAULT_DAILY_LOSS_STOP = 0.02
DEFAULT_MAX_LEVERAGE = 3.0

# Entry execution defaults for live mode
DEFAULT_LIMIT_ATR_OFFSET = 0.15
DEFAULT_LIMIT_TIMEOUT_SEC = 75
DEFAULT_FALLBACK_TO_MARKET = True
DEFAULT_MAX_MARKET_DEVIATION_PCT = 0.0015
DEFAULT_POSITION_IDX = 0

# Execution & costs
DEFAULT_INIT_CASH = 10_000.0
DEFAULT_SLIPPAGE = 0.0002
DEFAULT_FEE_MAKER = 0.0004
DEFAULT_FEE_TAKER = 0.0010
DEFAULT_FEE = DEFAULT_FEE_TAKER

# Closed trade journal
DEFAULT_TRADE_JOURNAL_PATH = "trade_journal.csv"
DEFAULT_CLOSED_PNL_SYNC_INTERVAL_SEC = 20
DEFAULT_TRADE_TELEGRAM_PREFIX = "TRADE_CLOSED"

# Optimization robustness controls
DEFAULT_OPT_MIN_TRADES = 20
DEFAULT_OPT_TARGET_TRADES = 40
DEFAULT_OPT_MIN_HALF_SIGNALS = 3


@dataclass(frozen=True, slots=True)
class StrategySettings:
    rsi_long_threshold: float = 38.0
    rsi_short_threshold: float = 62.0
    long_price_filter: str = "bb_lower"
    short_price_filter: str = "bb_upper"
    rr_ratio: float = 2.0
    atr_mult: float = 1.0
    use_ema_trend_filter: bool = False
    ema_length: int = 200
    atr_regime_min_pct: float = 0.0015
    atr_regime_max_pct: float = 0.0060


STRATEGY = StrategySettings()


console = Console()
SHUTDOWN = False
TELEGRAM_WARNING_SHOWN = False


def handle_shutdown(_sig: int, _frame: Any) -> None:
    global SHUTDOWN
    SHUTDOWN = True


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


@dataclass(slots=True)
class BotConfig:
    api_key: str
    api_secret: str
    telegram_bot_token: str
    telegram_chat_id: str
    symbol: str
    timeframe: str
    risk_per_trade: float
    daily_profit_stop: float
    daily_loss_stop: float
    max_leverage: float


def str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> BotConfig:
    load_dotenv()

    return BotConfig(
        api_key=os.getenv("BYBIT_API_KEY", ""),
        api_secret=os.getenv("BYBIT_API_SECRET", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        symbol=DEFAULT_SYMBOL,
        timeframe=DEFAULT_TIMEFRAME,
        risk_per_trade=DEFAULT_RISK_PER_TRADE,
        daily_profit_stop=DEFAULT_DAILY_PROFIT_STOP,
        daily_loss_stop=DEFAULT_DAILY_LOSS_STOP,
        max_leverage=DEFAULT_MAX_LEVERAGE,
    )


def create_exchange(config: BotConfig) -> ccxt.bybit:
    if not config.api_key or not config.api_secret:
        raise ValueError(
            "BYBIT_API_KEY/BYBIT_API_SECRET are required in .env for all bot modes"
        )

    exchange_params: dict[str, Any] = {
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "fetchCurrencies": False,
        },
        "apiKey": config.api_key,
        "secret": config.api_secret,
    }

    exchange = ccxt.bybit(exchange_params)
    return exchange


def send_telegram_alert(config: BotConfig, message: str) -> None:
    global TELEGRAM_WARNING_SHOWN
    if not config.telegram_bot_token or not config.telegram_chat_id:
        if not TELEGRAM_WARNING_SHOWN:
            console.print("[yellow]Telegram token/chat_id is missing, alerts are skipped.[/yellow]")
            TELEGRAM_WARNING_SHOWN = True
        return

    url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
    payload = {"chat_id": config.telegram_chat_id, "text": message}

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        console.print(f"[red]Telegram alert failed:[/red] {exc}")


def _ms_to_utc_iso(ms: Any) -> str:
    try:
        ms_int = int(float(ms))
        if ms_int <= 0:
            return ""
        return datetime.fromtimestamp(ms_int / 1000, tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _safe_pipe_value(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "/").replace("\n", " ").strip()


def format_trade_closed_telegram_line(record: dict[str, Any]) -> str:
    ordered_keys = [
        "trade_id",
        "logged_at_utc",
        "closed_time_utc",
        "symbol",
        "side",
        "qty",
        "avg_entry_price",
        "avg_exit_price",
        "closed_pnl",
        "exec_type",
        "order_id",
        "leverage",
    ]
    parts = [DEFAULT_TRADE_TELEGRAM_PREFIX]
    for key in ordered_keys:
        parts.append(f"{key}={_safe_pipe_value(record.get(key, ''))}")
    return "|".join(parts)


def init_trade_journal(path: str) -> set[str]:
    journal_path = Path(path)
    seen_ids: set[str] = set()

    if journal_path.exists():
        try:
            with journal_path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    trade_id = str(row.get("trade_id", "")).strip()
                    if trade_id:
                        seen_ids.add(trade_id)
        except Exception:
            pass
        return seen_ids

    with journal_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "trade_id",
                "logged_at_utc",
                "closed_time_utc",
                "symbol",
                "side",
                "qty",
                "avg_entry_price",
                "avg_exit_price",
                "closed_pnl",
                "exec_type",
                "order_id",
                "leverage",
            ]
        )
    return seen_ids


def sync_closed_trades_to_csv(
    exchange: ccxt.bybit,
    config: BotConfig,
    journal_path: str,
    seen_ids: set[str],
    limit: int = 50,
) -> int:
    method_name = "privateGetV5PositionClosedPnl"
    if not hasattr(exchange, method_name):
        return 0

    method = getattr(exchange, method_name)
    payload = {
        "category": "linear",
        "symbol": to_bybit_symbol(config.symbol),
        "limit": limit,
    }
    response = method(payload)
    rows = response.get("result", {}).get("list", []) if isinstance(response, dict) else []
    if not isinstance(rows, list) or not rows:
        return 0

    new_rows: list[list[Any]] = []
    telegram_lines: list[str] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for row in rows:
        if not isinstance(row, dict):
            continue
        trade_id = str(
            row.get("orderId")
            or row.get("execId")
            or row.get("updatedTime")
            or row.get("createdTime")
            or ""
        ).strip()
        if not trade_id or trade_id in seen_ids:
            continue

        try:
            closed_pnl = float(row.get("closedPnl", 0.0))
        except Exception:
            closed_pnl = 0.0

        # Keep only realized profit/loss trades (exclude exact breakeven records).
        if abs(closed_pnl) < 1e-12:
            continue

        record = {
            "trade_id": trade_id,
            "logged_at_utc": now_iso,
            "closed_time_utc": _ms_to_utc_iso(row.get("updatedTime") or row.get("createdTime")),
            "symbol": str(row.get("symbol", config.symbol)),
            "side": str(row.get("side", "")),
            "qty": str(row.get("qty", "")),
            "avg_entry_price": str(row.get("avgEntryPrice", "")),
            "avg_exit_price": str(row.get("avgExitPrice", "")),
            "closed_pnl": f"{closed_pnl:.8f}",
            "exec_type": str(row.get("execType", "")),
            "order_id": str(row.get("orderId", "")),
            "leverage": str(row.get("leverage", "")),
        }
        new_rows.append(
            [
                record["trade_id"],
                record["logged_at_utc"],
                record["closed_time_utc"],
                record["symbol"],
                record["side"],
                record["qty"],
                record["avg_entry_price"],
                record["avg_exit_price"],
                record["closed_pnl"],
                record["exec_type"],
                record["order_id"],
                record["leverage"],
            ]
        )
        telegram_lines.append(format_trade_closed_telegram_line(record))
        seen_ids.add(trade_id)

    if not new_rows:
        return 0

    with Path(journal_path).open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for item in new_rows:
            writer.writerow(item)

    for line in telegram_lines:
        send_telegram_alert(config, line)

    return len(new_rows)


def fetch_ohlcv_dataframe(
    exchange: ccxt.bybit, symbol: str, timeframe: str, limit: int
) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit must be > 0")

    max_batch = 1000
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    now_ms = exchange.milliseconds()
    since = now_ms - (limit + 200) * tf_ms

    data: list[list[float]] = []
    while len(data) < limit:
        batch_limit = min(max_batch, limit - len(data))
        batch = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=batch_limit)
        if not batch:
            break

        if data:
            last_ts = int(data[-1][0])
            batch = [row for row in batch if int(row[0]) > last_ts]
            if not batch:
                break

        data.extend(batch)
        since = int(batch[-1][0]) + tf_ms

        if len(batch) < batch_limit:
            break

    if len(data) > limit:
        data = data[-limit:]

    if not data:
        raise RuntimeError("No OHLCV data received from Bybit")

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.astype(float)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # RSI(10)
    result["rsi_10"] = ta.rsi(result["close"], length=10)

    # MACD(8, 21, 5)
    macd = ta.macd(result["close"], fast=8, slow=21, signal=5)
    if macd is None or macd.empty:
        raise RuntimeError("MACD calculation failed")
    result["macd"] = macd.iloc[:, 0]
    result["macd_signal"] = macd.iloc[:, 2]

    # Bollinger Bands(20)
    bb = ta.bbands(result["close"], length=20)
    if bb is None or bb.empty:
        raise RuntimeError("Bollinger Bands calculation failed")
    result["bb_lower"] = bb.iloc[:, 0]
    result["bb_mid"] = bb.iloc[:, 1]
    result["bb_upper"] = bb.iloc[:, 2]

    # ATR(14)
    result["atr_14"] = ta.atr(result["high"], result["low"], result["close"], length=14)

    # EMA trend filter
    result["ema_200"] = ta.ema(result["close"], length=200)

    # ATR regime as percentage of price
    result["atr_pct"] = (result["atr_14"] / result["close"]).clip(lower=0.0)

    return result.dropna().copy()


def generate_signals(
    df: pd.DataFrame,
    strategy: StrategySettings = STRATEGY,
) -> tuple[pd.Series, pd.Series]:
    macd_cross_up = (df["macd"] > df["macd_signal"]) & (
        df["macd"].shift(1) <= df["macd_signal"].shift(1)
    )
    macd_cross_down = (df["macd"] < df["macd_signal"]) & (
        df["macd"].shift(1) >= df["macd_signal"].shift(1)
    )

    long_entries = (
        (df["rsi_10"] < strategy.rsi_long_threshold)
        & macd_cross_up
        & (df["close"] > df[strategy.long_price_filter])
    )
    short_entries = (
        (df["rsi_10"] > strategy.rsi_short_threshold)
        & macd_cross_down
        & (df["close"] < df[strategy.short_price_filter])
    )

    # Volatility regime filter to avoid too-dead / too-chaotic conditions.
    atr_regime_ok = (df["atr_pct"] >= strategy.atr_regime_min_pct) & (
        df["atr_pct"] <= strategy.atr_regime_max_pct
    )
    long_entries = long_entries & atr_regime_ok
    short_entries = short_entries & atr_regime_ok

    # Trend alignment (optional)
    if strategy.use_ema_trend_filter:
        long_entries = long_entries & (df["close"] > df["ema_200"])
        short_entries = short_entries & (df["close"] < df["ema_200"])

    return long_entries.fillna(False), short_entries.fillna(False)


def build_risk_arrays(
    df: pd.DataFrame,
    risk_per_trade: float,
    rr_ratio: float = STRATEGY.rr_ratio,
    atr_mult: float = STRATEGY.atr_mult,
    max_leverage: float = 1.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Stop distance as fraction of current price
    sl_pct = ((df["atr_14"] * atr_mult) / df["close"]).clip(lower=1e-6)
    tp_pct = sl_pct * rr_ratio

    # Position sizing from risk model:
    # risk ~= position_fraction * sl_pct => position_fraction = risk/sl_pct
    size_pct = (risk_per_trade / sl_pct).clip(lower=0.0, upper=max_leverage)

    return sl_pct, tp_pct, size_pct


def run_vectorbt_backtest(
    df: pd.DataFrame,
    long_entries: pd.Series,
    short_entries: pd.Series,
    sl_pct: pd.Series,
    tp_pct: pd.Series,
    size_pct: pd.Series,
    slippage: float = 0.0002,
    fee: float = 0.0006,
    init_cash: float = 10_000.0,
) -> vbt.Portfolio:
    # Opposite signal closes current side.
    long_exits = short_entries
    short_exits = long_entries

    portfolio = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=size_pct,
        size_type="percent",
        sl_stop=sl_pct,
        tp_stop=tp_pct,
        fees=fee,
        slippage=slippage,
        init_cash=init_cash,
        freq="5min",
    )
    return portfolio


def monte_carlo_trades(
    trade_returns: np.ndarray,
    n_simulations: int = 2000,
    horizon_trades: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    if trade_returns.size == 0:
        return {
            "mc_median_return": 0.0,
            "mc_p5_return": 0.0,
            "mc_p95_return": 0.0,
        }

    rng = np.random.default_rng(seed)
    horizon = horizon_trades or trade_returns.size
    sampled = rng.choice(trade_returns, size=(n_simulations, horizon), replace=True)
    equity_paths = np.cumprod(1.0 + sampled, axis=1)
    terminal = equity_paths[:, -1] - 1.0

    return {
        "mc_median_return": float(np.median(terminal)),
        "mc_p5_return": float(np.percentile(terminal, 5)),
        "mc_p95_return": float(np.percentile(terminal, 95)),
    }


def walk_forward_analysis(
    df: pd.DataFrame,
    train_size: int = 1500,
    test_size: int = 500,
    step: int = 500,
    slippage: float = DEFAULT_SLIPPAGE,
    fee: float = DEFAULT_FEE,
    init_cash: float = DEFAULT_INIT_CASH,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
    strategy: StrategySettings = STRATEGY,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []

    for start in range(0, len(df) - train_size - test_size + 1, step):
        test_slice = df.iloc[start + train_size : start + train_size + test_size].copy()
        if test_slice.empty:
            continue

        long_entries, short_entries = generate_signals(test_slice, strategy=strategy)
        sl_pct, tp_pct, size_pct = build_risk_arrays(
            test_slice,
            risk_per_trade=risk_per_trade,
            rr_ratio=strategy.rr_ratio,
            atr_mult=strategy.atr_mult,
            max_leverage=max_leverage,
        )

        pf = run_vectorbt_backtest(
            test_slice,
            long_entries,
            short_entries,
            sl_pct,
            tp_pct,
            size_pct,
            slippage=slippage,
            fee=fee,
            init_cash=init_cash,
        )

        total_trades = int(pf.trades.count())
        win_rate = float(pf.trades.win_rate()) if total_trades > 0 else 0.0
        profit_factor = float(pf.trades.profit_factor()) if total_trades > 0 else 0.0
        total_return = float(pf.total_return())

        records.append(
            {
                "window_start": float(start + train_size),
                "window_end": float(start + train_size + test_size),
                "trades": float(total_trades),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_return": total_return,
            }
        )

    return pd.DataFrame(records)


def summarize_backtest(pf: vbt.Portfolio) -> dict[str, float]:
    trades = pf.trades
    total_trades = int(trades.count())

    if total_trades == 0:
        return {
            "total_return": float(pf.total_return()),
            "max_drawdown": float(pf.max_drawdown()),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0.0,
            "expectancy": 0.0,
        }

    # Expectancy per trade in R-like percentage form
    trade_returns = trades.returns.values
    expectancy = float(np.mean(trade_returns)) if trade_returns.size else 0.0

    return {
        "total_return": float(pf.total_return()),
        "max_drawdown": float(pf.max_drawdown()),
        "win_rate": float(trades.win_rate()),
        "profit_factor": float(trades.profit_factor()),
        "total_trades": float(total_trades),
        "expectancy": expectancy,
    }


def print_backtest_tables(metrics: dict[str, float], wfo_df: pd.DataFrame, mc: dict[str, float]) -> None:
    summary = Table(title="Backtest Summary", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", justify="left")
    summary.add_column("Value", justify="right")

    summary.add_row("Total Return", f"{metrics['total_return'] * 100:.2f}%")
    summary.add_row("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    summary.add_row("Win Rate", f"{metrics['win_rate'] * 100:.2f}%")
    summary.add_row("Profit Factor", f"{metrics['profit_factor']:.2f}")
    summary.add_row("Trades", f"{int(metrics['total_trades'])}")
    summary.add_row("Expectancy/trade", f"{metrics['expectancy'] * 100:.3f}%")

    console.print(summary)

    if not wfo_df.empty:
        wfo_table = Table(title="Walk-Forward (OOS)", box=box.SIMPLE)
        wfo_table.add_column("Window", justify="left")
        wfo_table.add_column("Trades", justify="right")
        wfo_table.add_column("WR", justify="right")
        wfo_table.add_column("PF", justify="right")
        wfo_table.add_column("Return", justify="right")

        for _, row in wfo_df.iterrows():
            wfo_table.add_row(
                f"{int(row['window_start'])}-{int(row['window_end'])}",
                str(int(row["trades"])),
                f"{row['win_rate'] * 100:.2f}%",
                f"{row['profit_factor']:.2f}",
                f"{row['total_return'] * 100:.2f}%",
            )
        console.print(wfo_table)

    mc_table = Table(title="Monte Carlo (Trade Bootstrap)", box=box.SIMPLE)
    mc_table.add_column("Metric", justify="left")
    mc_table.add_column("Value", justify="right")
    mc_table.add_row("Median terminal return", f"{mc['mc_median_return'] * 100:.2f}%")
    mc_table.add_row("5th percentile", f"{mc['mc_p5_return'] * 100:.2f}%")
    mc_table.add_row("95th percentile", f"{mc['mc_p95_return'] * 100:.2f}%")
    console.print(mc_table)


def show_plot(pf: vbt.Portfolio, title: str = "VectorBT Portfolio") -> None:
    try:
        fig = pf.plot(title=title)
        fig.show()
    except Exception as exc:
        console.print(f"[yellow]Plot skipped:[/yellow] {exc}")


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def ceil_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step) * step


def get_market_meta(exchange: ccxt.bybit, symbol: str) -> dict[str, float]:
    markets = exchange.load_markets()
    market = markets.get(symbol)
    if not market:
        raise RuntimeError(f"Market metadata not found for symbol: {symbol}")

    amount_step = market.get("precision", {}).get("amount", 3)
    price_step = market.get("precision", {}).get("price", 2)

    # Convert precision decimals to rough step values
    amount_tick = 10 ** (-amount_step) if isinstance(amount_step, int) else 0.001
    price_tick = 10 ** (-price_step) if isinstance(price_step, int) else 0.01

    min_amount = market.get("limits", {}).get("amount", {}).get("min", amount_tick)

    return {
        "amount_tick": float(amount_tick),
        "price_tick": float(price_tick),
        "min_amount": float(min_amount or amount_tick),
    }


def calculate_live_order_qty(
    equity_usdt: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float,
    max_leverage: float,
    min_amount: float,
    amount_tick: float,
) -> float:
    risk_cash = equity_usdt * risk_per_trade
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0

    raw_qty = risk_cash / stop_distance

    # Cap notional by leverage constraint
    max_notional = equity_usdt * max_leverage
    max_qty = max_notional / entry_price
    qty = min(raw_qty, max_qty)

    qty = floor_to_step(qty, amount_tick)
    if qty < min_amount:
        return 0.0
    return qty


def get_reference_price(exchange: ccxt.bybit, symbol: str) -> float | None:
    try:
        ticker = exchange.fetch_ticker(symbol)
        if not isinstance(ticker, dict):
            return None
        for field in ("last", "mark", "close"):
            value = ticker.get(field)
            if value is not None:
                return float(value)
    except Exception:
        return None
    return None


def extract_order_fill_price(order: dict[str, Any], fallback_price: float) -> float:
    for key in ("average", "avgPrice", "price", "lastTradeTimestamp"):
        value = order.get(key)
        if value is None:
            continue
        if key == "lastTradeTimestamp":
            continue
        try:
            parsed = float(value)
            if parsed > 0:
                return parsed
        except Exception:
            continue
    return fallback_price


def set_position_trading_stop(
    exchange: ccxt.bybit,
    config: BotConfig,
    take_profit: float,
    stop_loss: float,
) -> bool:
    method_name = "privatePostV5PositionTradingStop"
    if not hasattr(exchange, method_name):
        return False

    method = getattr(exchange, method_name)
    params = {
        "category": "linear",
        "symbol": to_bybit_symbol(config.symbol),
        "tpslMode": "Full",
        "positionIdx": DEFAULT_POSITION_IDX,
        "takeProfit": f"{take_profit:.6f}",
        "stopLoss": f"{stop_loss:.6f}",
    }
    try:
        response = method(params)
        return isinstance(response, dict) and str(response.get("retCode", "0")) in {"0", "None"}
    except Exception:
        return False


def place_entry_order_with_limit_first(
    exchange: ccxt.bybit,
    config: BotConfig,
    side: str,
    qty: float,
    signal_price: float,
    atr_value: float,
    price_tick: float,
) -> tuple[str, str | None, float | None]:
    limit_offset = atr_value * DEFAULT_LIMIT_ATR_OFFSET
    if side == "buy":
        raw_limit_price = signal_price - limit_offset
        limit_price = floor_to_step(raw_limit_price, price_tick)
    else:
        raw_limit_price = signal_price + limit_offset
        limit_price = ceil_to_step(raw_limit_price, price_tick)

    if limit_price <= 0:
        limit_price = signal_price

    try:
        limit_order = exchange.create_order(
            symbol=config.symbol,
            type="limit",
            side=side,
            amount=qty,
            price=limit_price,
            params={"timeInForce": "PostOnly"},
        )
        order_id = str(limit_order.get("id", ""))
        if not order_id:
            raise RuntimeError("Limit order id is missing")

        start_ts = time.time()
        while (time.time() - start_ts) < DEFAULT_LIMIT_TIMEOUT_SEC:
            state = exchange.fetch_order(order_id, config.symbol)
            status = str(state.get("status", "")).lower()
            if status in {"closed", "filled"}:
                fill_price = extract_order_fill_price(state, fallback_price=limit_price)
                return "limit_filled", order_id, fill_price
            if status in {"canceled", "rejected", "expired"}:
                break
            time.sleep(3)

        try:
            exchange.cancel_order(order_id, config.symbol)
        except Exception:
            pass

        if DEFAULT_FALLBACK_TO_MARKET:
            market_ref_price = get_reference_price(exchange, config.symbol) or signal_price
            deviation = abs(market_ref_price - signal_price) / max(signal_price, 1e-9)
            if deviation > DEFAULT_MAX_MARKET_DEVIATION_PCT:
                return "market_skipped_slippage", order_id, None

            market_order = exchange.create_order(
                symbol=config.symbol,
                type="market",
                side=side,
                amount=qty,
            )
            market_order_id = str(market_order.get("id", "N/A"))
            fill_price = extract_order_fill_price(market_order, fallback_price=market_ref_price)
            return "market_fallback", market_order_id, fill_price
        return "limit_timeout", order_id, None

    except Exception:
        if DEFAULT_FALLBACK_TO_MARKET:
            market_ref_price = get_reference_price(exchange, config.symbol) or signal_price
            deviation = abs(market_ref_price - signal_price) / max(signal_price, 1e-9)
            if deviation > DEFAULT_MAX_MARKET_DEVIATION_PCT:
                return "market_skipped_slippage", None, None

            market_order = exchange.create_order(
                symbol=config.symbol,
                type="market",
                side=side,
                amount=qty,
            )
            market_order_id = str(market_order.get("id", "N/A"))
            fill_price = extract_order_fill_price(market_order, fallback_price=market_ref_price)
            return "market_fallback", market_order_id, fill_price
        raise


def current_utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def to_bybit_symbol(symbol: str) -> str:
    # BTC/USDT:USDT -> BTCUSDT
    return symbol.split(":")[0].replace("/", "")


def to_bybit_interval(timeframe: str) -> str:
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe for Bybit WS: {timeframe}")
    return mapping[timeframe]


class BybitKlineStream:
    def __init__(self, config: BotConfig, symbol: str, timeframe: str) -> None:
        self.config = config
        self.symbol = to_bybit_symbol(symbol)
        self.interval = to_bybit_interval(timeframe)
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1000)
        self._thread: threading.Thread | None = None
        self._ws: websocket.WebSocketApp | None = None
        self._stop_event = threading.Event()
        self.connected = False
        self.last_message_ts = 0.0

    @property
    def ws_url(self) -> str:
        # Bybit public linear market data endpoint.
        return "wss://stream.bybit.com/v5/public/linear"

    @property
    def topic(self) -> str:
        return f"kline.{self.interval}.{self.symbol}"

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self.connected = True
        sub_message = {"op": "subscribe", "args": [self.topic]}
        ws.send(json.dumps(sub_message))
        console.print(f"[green]WS connected:[/green] {self.topic}")

    def _on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        self.last_message_ts = time.time()
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        topic = payload.get("topic", "")
        if not topic.startswith("kline"):
            return

        data_items = payload.get("data", [])
        if not isinstance(data_items, list):
            return

        for item in data_items:
            if not isinstance(item, dict):
                continue
            # Process only closed candles to avoid repainting
            if not bool(item.get("confirm", False)):
                continue

            kline = {
                "timestamp": pd.to_datetime(int(item["start"]), unit="ms", utc=True),
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": float(item["volume"]),
            }
            try:
                self.queue.put_nowait(kline)
            except queue.Full:
                _ = self.queue.get_nowait()
                self.queue.put_nowait(kline)

    def _on_error(self, _ws: websocket.WebSocketApp, error: Any) -> None:
        self.connected = False
        console.print(f"[yellow]WS error:[/yellow] {error}")

    def _on_close(
        self,
        _ws: websocket.WebSocketApp,
        _status_code: int | None,
        _msg: str | None,
    ) -> None:
        self.connected = False
        console.print("[yellow]WS connection closed.[/yellow]")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._ws.run_forever(ping_interval=20, ping_timeout=10)
            if self._stop_event.is_set():
                break
            time.sleep(2)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            self._ws.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)


def run_live_loop(config: BotConfig, exchange: ccxt.bybit, bars: int) -> None:
    console.print("[bold cyan]Starting live 5m loop...[/bold cyan]")
    console.print(
        f"Mode: LIVE EXECUTION | Symbol: {config.symbol} | Timeframe: {config.timeframe}"
    )

    market_meta = get_market_meta(exchange, config.symbol)
    history = fetch_ohlcv_dataframe(exchange, config.symbol, config.timeframe, limit=bars)
    latest_candles = deque(history.reset_index().to_dict("records"), maxlen=bars)
    stream = BybitKlineStream(config, config.symbol, config.timeframe)
    stream.start()

    seen_trade_ids = init_trade_journal(DEFAULT_TRADE_JOURNAL_PATH)
    last_journal_sync_ts = 0.0

    day_anchor = current_utc_day()
    day_start_equity: float | None = None
    halted_for_day = False

    while not SHUTDOWN:
        try:
            if current_utc_day() != day_anchor:
                day_anchor = current_utc_day()
                day_start_equity = None
                halted_for_day = False
                console.print("[green]New UTC day started: daily limits reset.[/green]")

            balance = exchange.fetch_balance()
            usdt_equity = float(balance.get("USDT", {}).get("total", 0.0))
            if usdt_equity <= 0:
                usdt_equity = float(balance.get("total", {}).get("USDT", 0.0))

            if day_start_equity is None and usdt_equity > 0:
                day_start_equity = usdt_equity

            if day_start_equity and day_start_equity > 0:
                day_pnl_pct = (usdt_equity / day_start_equity) - 1.0
                if day_pnl_pct >= config.daily_profit_stop or day_pnl_pct <= -config.daily_loss_stop:
                    halted_for_day = True
                    msg = (
                        f"Daily stop triggered: PnL={day_pnl_pct * 100:.2f}% "
                        f"(limits +{config.daily_profit_stop * 100:.2f}%/-{config.daily_loss_stop * 100:.2f}%)"
                    )
                    console.print(f"[yellow]{msg}[/yellow]")
                    send_telegram_alert(config, f"[BOT] {msg}")

            if halted_for_day:
                now_ts = time.time()
                if (now_ts - last_journal_sync_ts) >= DEFAULT_CLOSED_PNL_SYNC_INTERVAL_SEC:
                    added = sync_closed_trades_to_csv(
                        exchange=exchange,
                        config=config,
                        journal_path=DEFAULT_TRADE_JOURNAL_PATH,
                        seen_ids=seen_trade_ids,
                    )
                    if added > 0:
                        console.print(
                            f"[green]Trade journal updated:[/green] +{added} rows -> {DEFAULT_TRADE_JOURNAL_PATH}"
                        )
                    last_journal_sync_ts = now_ts
                time.sleep(30)
                continue

            kline = stream.queue.get(timeout=45)

            # Keep only the newest version for the same closed candle timestamp.
            if latest_candles and latest_candles[-1]["timestamp"] == kline["timestamp"]:
                latest_candles[-1] = kline
            else:
                latest_candles.append(kline)

            df = pd.DataFrame(list(latest_candles)).set_index("timestamp").sort_index().astype(float)
            df = add_indicators(df)
            long_entries, short_entries = generate_signals(df)

            last = df.iloc[-1]
            ts = df.index[-1]
            long_signal = bool(long_entries.iloc[-1])
            short_signal = bool(short_entries.iloc[-1])

            side = None
            if long_signal:
                side = "buy"
            elif short_signal:
                side = "sell"

            if side is None:
                console.print(f"[{ts}] No signal | Price={last['close']:.2f}")
            else:
                atr = float(last["atr_14"])
                entry_price = float(last["close"])

                qty = calculate_live_order_qty(
                    equity_usdt=usdt_equity,
                    entry_price=entry_price,
                    stop_price=sl_price,
                    risk_per_trade=config.risk_per_trade,
                    max_leverage=config.max_leverage,
                    min_amount=market_meta["min_amount"],
                    amount_tick=market_meta["amount_tick"],
                )

                if qty <= 0:
                    console.print("[yellow]Signal found, but qty <= 0 due to risk/precision constraints.[/yellow]")
                else:
                    est_sl = entry_price - atr if side == "buy" else entry_price + atr
                    est_tp = (
                        entry_price + atr * STRATEGY.rr_ratio
                        if side == "buy"
                        else entry_price - atr * STRATEGY.rr_ratio
                    )
                    msg = (
                        f"{ts} SIGNAL {side.upper()} {config.symbol}\n"
                        f"Entry(sig): {entry_price:.2f} | SL(est): {est_sl:.2f} | TP(est): {est_tp:.2f} | Qty: {qty}"
                    )
                    console.print(f"[bold green]{msg}[/bold green]")
                    send_telegram_alert(config, f"[BOT] {msg}")

                    execution_mode, order_id, fill_price = place_entry_order_with_limit_first(
                        exchange=exchange,
                        config=config,
                        side=side,
                        qty=qty,
                        signal_price=entry_price,
                        atr_value=atr,
                        price_tick=market_meta["price_tick"],
                    )

                    if fill_price is None:
                        console.print(
                            f"[yellow]Entry not executed ({execution_mode}); skipped due to guard/timeout.[/yellow]"
                        )
                        send_telegram_alert(
                            config,
                            f"[BOT] Entry not executed ({execution_mode}), signal was skipped.",
                        )
                        continue

                    sl_price = fill_price - atr if side == "buy" else fill_price + atr
                    tp_price = (
                        fill_price + atr * STRATEGY.rr_ratio
                        if side == "buy"
                        else fill_price - atr * STRATEGY.rr_ratio
                    )

                    stop_ok = set_position_trading_stop(
                        exchange=exchange,
                        config=config,
                        take_profit=tp_price,
                        stop_loss=sl_price,
                    )
                    stop_status = "TP/SL set" if stop_ok else "TP/SL set failed"

                    console.print(
                        f"[cyan]Order executed ({execution_mode}):[/cyan] {order_id} | "
                        f"fill={fill_price:.2f} | SL={sl_price:.2f} | TP={tp_price:.2f} | {stop_status}"
                    )
                    send_telegram_alert(
                        config,
                        f"[BOT] Executed ({execution_mode}) id={order_id} fill={fill_price:.2f} "
                        f"SL={sl_price:.2f} TP={tp_price:.2f} {stop_status}",
                    )

            # Optional WS health check with REST fallback refresh.
            if stream.last_message_ts and (time.time() - stream.last_message_ts) > 120:
                console.print("[yellow]WS stale detected, refreshing history via REST fallback...[/yellow]")
                history = fetch_ohlcv_dataframe(exchange, config.symbol, config.timeframe, limit=bars)
                latest_candles = deque(history.reset_index().to_dict("records"), maxlen=bars)

            now_ts = time.time()
            if (now_ts - last_journal_sync_ts) >= DEFAULT_CLOSED_PNL_SYNC_INTERVAL_SEC:
                added = sync_closed_trades_to_csv(
                    exchange=exchange,
                    config=config,
                    journal_path=DEFAULT_TRADE_JOURNAL_PATH,
                    seen_ids=seen_trade_ids,
                )
                if added > 0:
                    console.print(
                        f"[green]Trade journal updated:[/green] +{added} rows -> {DEFAULT_TRADE_JOURNAL_PATH}"
                    )
                last_journal_sync_ts = now_ts

        except queue.Empty:
            console.print("[yellow]No closed candle from WS yet, waiting...[/yellow]")
            if stream.last_message_ts and (time.time() - stream.last_message_ts) > 120:
                console.print("[yellow]WS timeout, rebuilding stream state from REST...[/yellow]")
                history = fetch_ohlcv_dataframe(exchange, config.symbol, config.timeframe, limit=bars)
                latest_candles = deque(history.reset_index().to_dict("records"), maxlen=bars)
        except ccxt.NetworkError as exc:
            console.print(f"[yellow]Network error:[/yellow] {exc}")
            time.sleep(10)
        except ccxt.ExchangeError as exc:
            console.print(f"[red]Exchange error:[/red] {exc}")
            time.sleep(10)
        except Exception as exc:
            console.print(f"[red]Unexpected error:[/red] {exc}")
            time.sleep(10)

    stream.stop()
    console.print("[bold]Graceful shutdown completed.[/bold]")


def run_backtest_command(
    config: BotConfig,
    exchange: ccxt.bybit,
    bars: int,
    with_plot: bool,
) -> int:
    console.print(f"[cyan]Fetching {bars} bars for {config.symbol} ({config.timeframe})...[/cyan]")
    df = fetch_ohlcv_dataframe(exchange, config.symbol, config.timeframe, limit=bars)
    df = add_indicators(df)
    if len(df) > 1:
        span_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600.0
        span_days = span_hours / 24.0
        console.print(
            f"[cyan]Sample span:[/cyan] {len(df)} bars | ~{span_hours:.1f}h (~{span_days:.2f} days)"
        )

    long_entries, short_entries = generate_signals(df, strategy=STRATEGY)
    sl_pct, tp_pct, size_pct = build_risk_arrays(
        df,
        risk_per_trade=config.risk_per_trade,
        rr_ratio=STRATEGY.rr_ratio,
        atr_mult=STRATEGY.atr_mult,
        max_leverage=config.max_leverage,
    )

    pf = run_vectorbt_backtest(
        df,
        long_entries,
        short_entries,
        sl_pct,
        tp_pct,
        size_pct,
        slippage=DEFAULT_SLIPPAGE,
        fee=DEFAULT_FEE,
    )

    metrics = summarize_backtest(pf)

    trade_returns = pf.trades.returns.values
    mc = monte_carlo_trades(trade_returns, n_simulations=2000)

    wfo_df = walk_forward_analysis(
        df,
        train_size=1500,
        test_size=500,
        step=500,
        slippage=DEFAULT_SLIPPAGE,
        fee=DEFAULT_FEE,
        init_cash=DEFAULT_INIT_CASH,
        risk_per_trade=config.risk_per_trade,
        max_leverage=config.max_leverage,
        strategy=STRATEGY,
    )

    print_backtest_tables(metrics, wfo_df, mc)

    # Soft checks for target profile
    target_wr = 0.60
    target_pf = 1.50
    if metrics["win_rate"] < target_wr or metrics["profit_factor"] < target_pf:
        console.print(
            "[yellow]Target check not met on this sample: "
            f"WR={metrics['win_rate'] * 100:.2f}% (target > {target_wr * 100:.0f}%), "
            f"PF={metrics['profit_factor']:.2f} (target > {target_pf:.2f}).[/yellow]"
        )
    else:
        console.print("[bold green]Target profile achieved on this sample (WR/PF).[/bold green]")

    if with_plot:
        show_plot(pf, title=f"{config.symbol} {config.timeframe} Strategy")

    return 0


def iter_strategy_candidates() -> list[StrategySettings]:
    candidates: list[StrategySettings] = []
    for rsi_long in [28.0, 30.0, 33.0, 35.0, 38.0]:
        for rsi_short in [62.0, 65.0, 68.0, 70.0, 72.0]:
            for long_filter in ["bb_lower", "bb_mid"]:
                for short_filter in ["bb_upper", "bb_mid"]:
                    for atr_mult in [0.8, 1.0, 1.2, 1.4]:
                        for use_trend in [True, False]:
                            for atr_min, atr_max in [(0.0012, 0.0055), (0.0015, 0.0060), (0.0018, 0.0065)]:
                                candidates.append(
                                    StrategySettings(
                                        rsi_long_threshold=rsi_long,
                                        rsi_short_threshold=rsi_short,
                                        long_price_filter=long_filter,
                                        short_price_filter=short_filter,
                                        rr_ratio=2.0,
                                        atr_mult=atr_mult,
                                        use_ema_trend_filter=use_trend,
                                        ema_length=200,
                                        atr_regime_min_pct=atr_min,
                                        atr_regime_max_pct=atr_max,
                                    )
                                )
    return candidates


def score_metrics(metrics: dict[str, float], trades: int) -> float:
    wr = float(metrics["win_rate"])
    pf_val = float(metrics["profit_factor"])
    ret_val = float(metrics["total_return"])
    dd = abs(float(metrics["max_drawdown"]))

    wr_component = max(0.0, (wr - 0.45) / 0.55)
    pf_component = max(0.0, pf_val)
    trades_component = min(1.0, trades / DEFAULT_OPT_TARGET_TRADES)
    drawdown_penalty = 1.0 + dd * 4.0
    return (
        (pf_component**1.35)
        * (1.0 + wr_component)
        * (1.0 + max(0.0, ret_val))
        * trades_component
    ) / drawdown_penalty


def signal_distribution(entries: pd.Series, shorts: pd.Series) -> tuple[int, int]:
    signal_mask = (entries | shorts).fillna(False)
    split_idx = int(len(signal_mask) * 0.5)
    return int(signal_mask.iloc[:split_idx].sum()), int(signal_mask.iloc[split_idx:].sum())


def evaluate_strategy(
    df: pd.DataFrame,
    strategy: StrategySettings,
    config: BotConfig,
) -> tuple[dict[str, float], int, int, int]:
    long_entries, short_entries = generate_signals(df, strategy=strategy)
    h1, h2 = signal_distribution(long_entries, short_entries)

    sl_pct, tp_pct, size_pct = build_risk_arrays(
        df,
        risk_per_trade=config.risk_per_trade,
        rr_ratio=strategy.rr_ratio,
        atr_mult=strategy.atr_mult,
        max_leverage=config.max_leverage,
    )
    pf = run_vectorbt_backtest(
        df,
        long_entries,
        short_entries,
        sl_pct,
        tp_pct,
        size_pct,
        slippage=DEFAULT_SLIPPAGE,
        fee=DEFAULT_FEE,
        init_cash=DEFAULT_INIT_CASH,
    )
    metrics = summarize_backtest(pf)
    trades = int(metrics["total_trades"])
    return metrics, trades, h1, h2


def run_optimize_command(config: BotConfig, exchange: ccxt.bybit, bars: int, top_n: int) -> int:
    console.print(f"[cyan]Optimizing on {bars} bars for {config.symbol} ({config.timeframe})...[/cyan]")
    df = fetch_ohlcv_dataframe(exchange, config.symbol, config.timeframe, limit=bars)
    df = add_indicators(df)

    results: list[dict[str, float | str]] = []
    for strategy in iter_strategy_candidates():
        metrics, trades, first_half_signals, second_half_signals = evaluate_strategy(df, strategy, config)

        if trades < DEFAULT_OPT_MIN_TRADES:
            continue
        if (
            first_half_signals < DEFAULT_OPT_MIN_HALF_SIGNALS
            or second_half_signals < DEFAULT_OPT_MIN_HALF_SIGNALS
        ):
            continue

        score = score_metrics(metrics, trades)
        results.append(
            {
                "rsi_long": strategy.rsi_long_threshold,
                "rsi_short": strategy.rsi_short_threshold,
                "long_filter": strategy.long_price_filter,
                "short_filter": strategy.short_price_filter,
                "atr_mult": strategy.atr_mult,
                "trend": "EMA200" if strategy.use_ema_trend_filter else "none",
                "atr_band": f"{strategy.atr_regime_min_pct:.4f}-{strategy.atr_regime_max_pct:.4f}",
                "trades": float(trades),
                "signals_h1": float(first_half_signals),
                "signals_h2": float(second_half_signals),
                "wr": float(metrics["win_rate"]),
                "pf": float(metrics["profit_factor"]),
                "ret": float(metrics["total_return"]),
                "dd": abs(float(metrics["max_drawdown"])),
                "score": score,
            }
        )

    if not results:
        console.print("[yellow]No valid optimization candidates found.[/yellow]")
        return 0

    ranked = sorted(results, key=lambda row: float(row["score"]), reverse=True)
    table = Table(title="Optimization Top Candidates", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right")
    table.add_column("RSI", justify="left")
    table.add_column("Filters", justify="left")
    table.add_column("Trend", justify="left")
    table.add_column("ATR Band", justify="left")
    table.add_column("ATR", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("H1/H2", justify="right")
    table.add_column("WR", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("DD", justify="right")
    table.add_column("Return", justify="right")

    for idx, row in enumerate(ranked[:top_n], start=1):
        table.add_row(
            str(idx),
            f"<{row['rsi_long']:.0f} / >{row['rsi_short']:.0f}",
            f"{row['long_filter']} / {row['short_filter']}",
            str(row["trend"]),
            str(row["atr_band"]),
            f"{row['atr_mult']:.1f}",
            f"{int(row['trades'])}",
            f"{int(row['signals_h1'])}/{int(row['signals_h2'])}",
            f"{row['wr'] * 100:.2f}%",
            f"{row['pf']:.2f}",
            f"{row['dd'] * 100:.2f}%",
            f"{row['ret'] * 100:.2f}%",
        )
    console.print(table)

    best = ranked[0]
    console.print(
        "[green]Best candidate:[/green] "
        f"RSI<{best['rsi_long']:.0f}/>{best['rsi_short']:.0f}, "
        f"filters {best['long_filter']}/{best['short_filter']}, "
        f"trend {best['trend']}, atr-band {best['atr_band']}, "
        f"ATRx{best['atr_mult']:.1f}, WR={best['wr'] * 100:.2f}%, PF={best['pf']:.2f}"
    )
    return 0


def run_optimize_wfo_command(
    config: BotConfig,
    exchange: ccxt.bybit,
    bars: int,
    train_size: int,
    test_size: int,
    step: int,
    top_n: int,
) -> int:
    console.print(
        f"[cyan]WFO optimize on {bars} bars | train={train_size}, test={test_size}, step={step}[/cyan]"
    )
    df = fetch_ohlcv_dataframe(exchange, config.symbol, config.timeframe, limit=bars)
    df = add_indicators(df)

    candidates = iter_strategy_candidates()
    fold_rows: list[dict[str, float | str]] = []

    for start in range(0, len(df) - train_size - test_size + 1, step):
        train_df = df.iloc[start : start + train_size].copy()
        test_df = df.iloc[start + train_size : start + train_size + test_size].copy()
        if train_df.empty or test_df.empty:
            continue

        effective_min_trades = max(8, min(DEFAULT_OPT_MIN_TRADES, int(len(train_df) * 0.005)))
        effective_min_half_signals = max(2, min(DEFAULT_OPT_MIN_HALF_SIGNALS, effective_min_trades // 4))

        best_train: dict[str, Any] | None = None
        for strategy in candidates:
            train_metrics, train_trades, train_h1, train_h2 = evaluate_strategy(train_df, strategy, config)
            if train_trades < effective_min_trades:
                continue
            if train_h1 < effective_min_half_signals or train_h2 < effective_min_half_signals:
                continue

            train_score = score_metrics(train_metrics, train_trades)
            row = {
                "strategy": strategy,
                "train_score": train_score,
                "train_trades": train_trades,
                "train_wr": float(train_metrics["win_rate"]),
                "train_pf": float(train_metrics["profit_factor"]),
            }
            if best_train is None or float(row["train_score"]) > float(best_train["train_score"]):
                best_train = row

        if best_train is None:
            continue

        strategy = best_train["strategy"]
        test_metrics, test_trades, test_h1, test_h2 = evaluate_strategy(test_df, strategy, config)
        fold_rows.append(
            {
                "fold": float(len(fold_rows) + 1),
                "start": float(start),
                "train_score": float(best_train["train_score"]),
                "train_trades": float(best_train["train_trades"]),
                "test_trades": float(test_trades),
                "test_wr": float(test_metrics["win_rate"]),
                "test_pf": float(test_metrics["profit_factor"]),
                "test_ret": float(test_metrics["total_return"]),
                "test_dd": abs(float(test_metrics["max_drawdown"])),
                "test_h1": float(test_h1),
                "test_h2": float(test_h2),
                "strategy": (
                    f"RSI<{strategy.rsi_long_threshold:.0f}/>{strategy.rsi_short_threshold:.0f} "
                    f"{strategy.long_price_filter}/{strategy.short_price_filter} "
                    f"trend={'EMA200' if strategy.use_ema_trend_filter else 'none'} "
                    f"atr_band={strategy.atr_regime_min_pct:.4f}-{strategy.atr_regime_max_pct:.4f} "
                    f"atrx={strategy.atr_mult:.1f}"
                ),
            }
        )

    if not fold_rows:
        console.print("[yellow]No WFO folds produced valid candidates. Increase bars or loosen constraints.[/yellow]")
        return 0

    fold_df = pd.DataFrame(fold_rows)
    summary = Table(title="WFO Out-of-Sample Summary", box=box.SIMPLE_HEAVY)
    summary.add_column("Folds", justify="right")
    summary.add_column("Avg Trades", justify="right")
    summary.add_column("Avg WR", justify="right")
    summary.add_column("Avg PF", justify="right")
    summary.add_column("Avg Return", justify="right")
    summary.add_column("Avg DD", justify="right")
    summary.add_row(
        str(len(fold_df)),
        f"{fold_df['test_trades'].mean():.1f}",
        f"{fold_df['test_wr'].mean() * 100:.2f}%",
        f"{fold_df['test_pf'].replace([np.inf, -np.inf], np.nan).mean():.2f}",
        f"{fold_df['test_ret'].mean() * 100:.2f}%",
        f"{fold_df['test_dd'].mean() * 100:.2f}%",
    )
    console.print(summary)

    table = Table(title="WFO Fold Details (Top by OOS PF)", box=box.SIMPLE)
    table.add_column("Fold", justify="right")
    table.add_column("Test Trades", justify="right")
    table.add_column("Test WR", justify="right")
    table.add_column("Test PF", justify="right")
    table.add_column("Test Return", justify="right")
    table.add_column("Strategy", justify="left")

    ranked = fold_df.sort_values(by=["test_pf", "test_wr", "test_ret"], ascending=False)
    for _, row in ranked.head(top_n).iterrows():
        table.add_row(
            str(int(row["fold"])),
            str(int(row["test_trades"])),
            f"{row['test_wr'] * 100:.2f}%",
            f"{row['test_pf']:.2f}",
            f"{row['test_ret'] * 100:.2f}%",
            str(row["strategy"]),
        )
    console.print(table)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bybit BTCUSDT 5m scalping bot (VectorBT + CCXT)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Run vectorized backtest")
    backtest.add_argument("--bars", type=int, default=5000, help="Number of OHLCV bars to fetch")
    backtest.add_argument("--plot", action="store_true", help="Show Plotly portfolio chart")

    optimize = subparsers.add_parser("optimize", help="Run parameter optimization")
    optimize.add_argument("--bars", type=int, default=5000, help="Number of OHLCV bars to fetch")
    optimize.add_argument("--top", type=int, default=10, help="How many top candidates to print")

    optimize_wfo = subparsers.add_parser("optimize_wfo", help="Run walk-forward out-of-sample optimization")
    optimize_wfo.add_argument("--bars", type=int, default=10000, help="Number of OHLCV bars to fetch")
    optimize_wfo.add_argument("--train", type=int, default=3000, help="Train window size")
    optimize_wfo.add_argument("--test", type=int, default=1000, help="Test window size")
    optimize_wfo.add_argument("--step", type=int, default=1000, help="Window step size")
    optimize_wfo.add_argument("--top", type=int, default=10, help="How many top folds to print")

    live = subparsers.add_parser("live", help="Run 5m live loop")
    live.add_argument("--bars", type=int, default=500, help="Bars to pull each loop")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    exchange = create_exchange(config)

    if args.command == "backtest":
        return run_backtest_command(config, exchange, bars=args.bars, with_plot=args.plot)

    if args.command == "optimize":
        return run_optimize_command(config, exchange, bars=args.bars, top_n=args.top)

    if args.command == "optimize_wfo":
        return run_optimize_wfo_command(
            config,
            exchange,
            bars=args.bars,
            train_size=args.train,
            test_size=args.test,
            step=args.step,
            top_n=args.top,
        )

    if args.command == "live":
        run_live_loop(config, exchange, bars=args.bars)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
