"""
Automatischer Stock Screener
Sucht täglich die besten handelbaren Aktien & ETFs von Alpaca
und bewertet sie nach einem Score-System.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional
import pytz
import pandas as pd
import broker
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus


# Feste Basis-Symbole die immer gescannt werden
BASE_SYMBOLS = ["AAPL", "GLD", "SPY"]

# Aktuelle Top-Kandidaten vom Screener
screener_results = []
last_screen_time = None


def get_all_assets() -> list:
    """Holt alle handelbaren US-Aktien und ETFs von Alpaca."""
    client = broker.get_client()
    req    = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = client.get_all_assets(req)
    return [a for a in assets if a.tradable and a.fractionable is not None]


def get_bars_df(symbol: str, days: int = 20) -> Optional[pd.DataFrame]:
    """Holt Tagesdaten für ein Symbol."""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        dc    = broker.get_data_client()
        end   = datetime.now(pytz.UTC)
        start = end - timedelta(days=days + 5)
        req   = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=days,
        )
        bars = dc.get_stock_bars(req)
        if symbol not in bars or len(bars[symbol]) < 10:
            return None

        rows = [{"close": b.close, "high": b.high,
                 "low": b.low, "volume": b.volume} for b in bars[symbol]]
        return pd.DataFrame(rows)
    except Exception:
        return None


def calculate_score(df: pd.DataFrame, symbol: str) -> Optional[dict]:
    """
    Berechnet einen Score 0-100 für ein Symbol.
    Höherer Score = besseres Trade-Kandidat.
    """
    try:
        if len(df) < 15:
            return None

        close  = df["close"].iloc[-1]
        volume = df["volume"].iloc[-1]

        # Preis-Filter: nur zwischen 5$ und 1000$
        if close < 5 or close > 1000:
            return None

        # Volumen-Filter: mind. 500k tägliches Volumen
        avg_vol = df["volume"].mean()
        if avg_vol < 500_000:
            return None

        # RSI berechnen
        delta = df["close"].diff()
        gain  = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
        loss  = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rsi   = (100 - (100 / (1 + gain / loss))).iloc[-1]

        # EMA200 (Trend-Filter)
        ema50  = df["close"].ewm(span=min(50, len(df)), adjust=False).mean().iloc[-1]
        ema200 = df["close"].ewm(span=min(200, len(df)), adjust=False).mean().iloc[-1]

        # ATR
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        atr_pct = atr / close * 100

        # Volumen-Ratio (heute vs. Durchschnitt)
        vol_ratio = volume / avg_vol

        # Score berechnen (0-100)
        score = 0

        # RSI-Signal (30 Punkte): Überverkauft = BUY Chance
        if rsi < 30:
            score += 30
            signal = "buy"
        elif rsi < 40:
            score += 20
            signal = "buy"
        elif rsi > 70:
            score += 30
            signal = "sell"
        elif rsi > 60:
            score += 20
            signal = "sell"
        else:
            return None  # Kein klares Signal

        # Trend-Bestätigung (25 Punkte)
        if signal == "buy" and close > ema50:
            score += 25
        elif signal == "sell" and close < ema50:
            score += 25
        elif signal == "buy" and close > ema200:
            score += 10
        elif signal == "sell" and close < ema200:
            score += 10

        # Volumen-Boost (25 Punkte)
        if vol_ratio > 2.0:
            score += 25
        elif vol_ratio > 1.5:
            score += 15
        elif vol_ratio > 1.0:
            score += 10

        # Volatilität (20 Punkte): moderate ATR bevorzugt
        if 1.0 < atr_pct < 5.0:
            score += 20
        elif atr_pct < 1.0:
            score += 5  # Zu wenig Bewegung

        return {
            "symbol":    symbol,
            "score":     round(score, 1),
            "signal":    signal,
            "price":     round(close, 2),
            "rsi":       round(rsi, 1),
            "vol_ratio": round(vol_ratio, 2),
            "atr_pct":   round(atr_pct, 2),
            "trend":     "bullisch" if close > ema50 else "bärisch",
        }
    except Exception:
        return None


def run_screener(push_fn=None, max_results: int = 10) -> list:
    """
    Hauptfunktion: Scannt alle Aktien und gibt Top-Kandidaten zurück.
    """
    global screener_results, last_screen_time

    print("[Screener] Starte täglichen Markt-Scan...")
    if push_fn:
        push_fn("screener", {"status": "Scanne Markt...", "results": [], "progress": 0})

    try:
        assets = get_all_assets()
    except Exception as e:
        print(f"[Screener] Fehler beim Laden der Assets: {e}")
        return []

    # Auf liquide, bekannte Aktien beschränken (S&P500 ähnlich)
    # Filter: nur Aktien mit bestimmten Eigenschaften
    candidates = []
    total      = min(len(assets), 500)  # Max 500 prüfen
    checked    = 0

    for asset in assets[:500]:
        symbol = asset.symbol
        # Symbole mit Sonderzeichen überspringen
        if not symbol.isalpha() or len(symbol) > 5:
            continue

        checked += 1
        if checked % 50 == 0:
            progress = int(checked / total * 100)
            print(f"[Screener] {checked}/{total} geprüft...")
            if push_fn:
                push_fn("screener", {
                    "status":   f"Scanne... {checked}/{total} Aktien geprüft",
                    "results":  [],
                    "progress": progress,
                })

        df = get_bars_df(symbol, days=20)
        if df is None:
            continue

        result = calculate_score(df, symbol)
        if result and result["score"] >= 50:
            candidates.append(result)

        time.sleep(0.05)  # Rate-Limit schonen

    # Nach Score sortieren
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:max_results]

    screener_results = top
    last_screen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"[Screener] Fertig! {len(top)} Top-Kandidaten gefunden:")
    for r in top:
        print(f"  {r['symbol']:6} Score:{r['score']:5} RSI:{r['rsi']:5} Signal:{r['signal']}")

    if push_fn:
        push_fn("screener", {
            "status":    f"Fertig — {len(top)} Kandidaten gefunden",
            "results":   top,
            "progress":  100,
            "last_scan": last_screen_time,
        })

    return top


def get_active_symbols() -> list:
    """Gibt Basis-Symbole + Top Screener-Kandidaten zurück."""
    dynamic = [r["symbol"] for r in screener_results]
    combined = list(dict.fromkeys(BASE_SYMBOLS + dynamic))
    return combined
