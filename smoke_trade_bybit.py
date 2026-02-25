#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import ccxt
from dotenv import load_dotenv
import os


@dataclass(slots=True)
class Config:
    api_key: str
    api_secret: str
    symbol: str
    wait_sec: float


def load_config(symbol: str, wait_sec: float) -> Config:
    load_dotenv()
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")
    return Config(api_key=api_key, api_secret=api_secret, symbol=symbol, wait_sec=wait_sec)


def create_exchange(cfg: Config) -> ccxt.bybit:
    return ccxt.bybit(
        {
            "enableRateLimit": True,
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "options": {"defaultType": "swap", "fetchCurrencies": False},
        }
    )


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return int(value / step) * step


def get_min_amount(exchange: ccxt.bybit, symbol: str) -> tuple[float, float]:
    markets = exchange.load_markets()
    market = markets.get(symbol)
    if not market:
        raise RuntimeError(f"Symbol not found: {symbol}")

    precision_amount = market.get("precision", {}).get("amount", 0)
    amount_tick = 10 ** (-precision_amount) if isinstance(precision_amount, int) else 1.0
    min_amount = market.get("limits", {}).get("amount", {}).get("min")
    if min_amount is None:
        min_amount = amount_tick
    return float(min_amount), float(amount_tick)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def get_open_position_qty(exchange: ccxt.bybit, symbol: str) -> float:
    try:
        positions = exchange.fetch_positions([symbol])
    except Exception:
        return 0.0
    if not isinstance(positions, list):
        return 0.0

    for pos in positions:
        if str(pos.get("symbol", "")) != symbol:
            continue
        contracts = safe_float(pos.get("contracts", 0.0))
        if contracts > 0:
            return contracts
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bybit real-order smoke test (open and close minimal position)")
    parser.add_argument("--symbol", default="1000PEPE/USDT:USDT", help="Linear perpetual symbol")
    parser.add_argument("--wait-sec", type=float, default=2.0, help="Seconds to wait before close")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually place orders. Without this flag script does only readiness checks.",
    )
    args = parser.parse_args()

    cfg = load_config(args.symbol, args.wait_sec)
    exchange = create_exchange(cfg)

    print(f"[CHECK] account + market for {cfg.symbol}")
    balance = exchange.fetch_balance()
    usdt_total = safe_float((balance.get("USDT") or {}).get("total", 0.0))
    min_amount, amount_tick = get_min_amount(exchange, cfg.symbol)
    ticker = exchange.fetch_ticker(cfg.symbol)
    last_price = safe_float(ticker.get("last", 0.0))
    min_notional_est = min_amount * last_price

    print(f"[OK] USDT total: {usdt_total:.4f}")
    print(f"[OK] min amount: {min_amount}, amount tick: {amount_tick}")
    print(f"[OK] last price: {last_price:.8f}, min notional est: {min_notional_est:.6f} USDT")

    if not args.execute:
        print("[DRY-RUN] Checks passed. Add --execute to open and close a real position.")
        return 0

    qty = floor_to_step(min_amount, amount_tick)
    if qty < min_amount:
        qty = min_amount

    print(f"[TRADE] Opening long market: qty={qty}")
    open_order = exchange.create_order(cfg.symbol, "market", "buy", qty)
    open_id = str(open_order.get("id", ""))
    print(f"[OK] Opened order id={open_id}")

    if cfg.wait_sec > 0:
        time.sleep(cfg.wait_sec)

    close_qty = get_open_position_qty(exchange, cfg.symbol)
    if close_qty <= 0:
        close_qty = safe_float(open_order.get("filled", 0.0), 0.0)
    if close_qty <= 0:
        close_qty = qty
    close_qty = floor_to_step(close_qty, amount_tick)
    if close_qty <= 0:
        close_qty = qty

    print(f"[TRADE] Closing long market reduceOnly: qty={close_qty}")
    close_order = exchange.create_order(
        cfg.symbol,
        "market",
        "sell",
        close_qty,
        params={"reduceOnly": True, "positionIdx": 0},
    )
    close_id = str(close_order.get("id", ""))
    print(f"[OK] Closed order id={close_id}")

    left_qty = get_open_position_qty(exchange, cfg.symbol)
    print(f"[RESULT] Remaining position qty: {left_qty}")
    if left_qty > 0:
        print("[WARN] Position still open. Re-run close manually in Bybit UI.")
        return 2

    print("[DONE] Smoke trade cycle completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
