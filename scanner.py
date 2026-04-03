"""
Alpaca Market Scanner
Läuft alle 5 Minuten, berechnet RSI/EMA/ATR/Volumen direkt aus Alpaca-Daten
und gibt Signale zurück — kein TradingView nötig.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional
import pytz
import pandas as pd
import xgboost as xgb
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import broker


def get_bars_df(symbol: str, timeframe: TimeFrame, limit: int = 250) -> pd.DataFrame:
    """Holt Bars von Alpaca als DataFrame."""
    dc = broker.get_data_client()
    end   = datetime.now(pytz.UTC)
    start = end - timedelta(days=60)
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=limit,
    )
    bars = dc.get_stock_bars(req)
    if symbol not in bars or not bars[symbol]:
        return pd.DataFrame()

    rows = [{
        "open":   b.open,
        "high":   b.high,
        "low":    b.low,
        "close":  b.close,
        "volume": b.volume,
    } for b in bars[symbol]]
    return pd.DataFrame(rows)


def calculate_indicators(df: pd.DataFrame, rsi_period: int = 14,
                          ema_fast: int = 50, ema_slow: int = 200) -> pd.DataFrame:
    """Berechnet RSI, EMA50, EMA200, ATR, Volumen-Ratio."""
    df = df.copy()

    # RSI
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).ewm(span=rsi_period, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(span=rsi_period, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # EMAs
    df["ema_fast"] = df["close"].ewm(span=ema_fast,  adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    # ATR
    df["tr"]  = (df["high"] - df["low"]).rolling(14).mean()
    df["atr"] = df["tr"]

    # Volumen-Ratio
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df.dropna()


def check_signal(df: pd.DataFrame, cfg: dict) -> Optional[str]:
    """
    Prüft ob ein BUY oder SELL Signal vorliegt.
    Gibt 'buy', 'sell' oder None zurück.
    """
    if len(df) < 2:
        return None

    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    rsi_os = cfg["rsi"]["oversold"]
    rsi_ob = cfg["rsi"]["overbought"]

    # BUY: RSI kreuzt nach oben über Oversold-Level + Kurs über EMA200
    buy = (prev["rsi"] < rsi_os and last["rsi"] >= rsi_os
           and last["close"] > last["ema_slow"])

    # SELL: RSI kreuzt nach unten unter Overbought-Level + Kurs unter EMA200
    sell = (prev["rsi"] > rsi_ob and last["rsi"] <= rsi_ob
            and last["close"] < last["ema_slow"])

    if buy:
        return "buy"
    if sell:
        return "sell"
    return None


def check_multiframe(symbol: str, signal: str) -> tuple[bool, str]:
    """4H Bestätigung: RSI-Trend muss zum Signal passen."""
    df_4h = get_bars_df(symbol, TimeFrame.Hour, limit=120)
    if df_4h.empty or len(df_4h) < 20:
        return True, ""

    # Simuliere 4H aus 1H Bars
    df_4h = df_4h.iloc[::4].reset_index(drop=True)
    df_4h = calculate_indicators(df_4h)
    if df_4h.empty:
        return True, ""

    rsi_4h = df_4h.iloc[-1]["rsi"]
    if signal == "buy"  and rsi_4h < 60:
        return True, ""
    if signal == "sell" and rsi_4h > 40:
        return True, ""
    return False, f"4H RSI={rsi_4h:.1f} bestätigt {signal.upper()} nicht"


class Scanner:
    def __init__(self, execute_fn, push_fn):
        """
        execute_fn: Funktion die einen Trade ausführt
        push_fn:    Funktion die SSE-Events ans Dashboard schickt
        """
        self.execute_fn = execute_fn
        self.push_fn    = push_fn
        self.models     = {}
        self.last_scan  = None
        self.last_signals = {}

        # ML-Modelle laden
        for sym in ("AAPL", "GLD", "SPY"):
            m = xgb.XGBClassifier()
            try:
                m.load_model(f"model_{sym}.json")
                self.models[sym] = m
            except Exception:
                pass

    def ml_ok(self, symbol: str, row, signal: str) -> bool:
        model = self.models.get(symbol)
        if not model:
            return True
        ema200_dist = (row["close"] - row["ema_slow"]) / row["ema_slow"] * 100
        ema50_dist  = (row["close"] - row["ema_fast"]) / row["ema_fast"] * 100
        df_in = pd.DataFrame(
            [[row["rsi"], ema200_dist, ema50_dist, row["atr"], row["vol_ratio"]]],
            columns=["RSI", "EMA200_dist", "EMA50_dist", "ATR", "Vol_ratio"]
        )
        pred = model.predict(df_in)[0]
        return (signal == "buy" and pred == 1) or (signal == "sell" and pred == 0)

    def scan_symbol(self, symbol: str, cfg: dict):
        """Scannt ein Symbol und führt ggf. einen Trade aus."""
        print(f"[Scanner] Scanne {symbol}...")

        df = get_bars_df(symbol, TimeFrame.Minute, limit=250)
        if df.empty:
            print(f"[Scanner] Keine Daten für {symbol}")
            return

        df = calculate_indicators(df)
        if df.empty:
            return

        signal = check_signal(df, cfg)
        if not signal:
            return

        last = df.iloc[-1]
        print(f"[Scanner] Signal: {signal.upper()} {symbol} | RSI={last['rsi']:.1f}")

        # Gleiches Signal nicht doppelt senden
        if self.last_signals.get(symbol) == signal:
            print(f"[Scanner] Signal {signal} für {symbol} bereits verarbeitet")
            return

        # Volumen prüfen
        vol_threshold = cfg.get("volume_threshold", 0.8)
        if last["vol_ratio"] < vol_threshold:
            reason = f"Volumen zu niedrig ({last['vol_ratio']:.2f}x Durchschnitt)"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            return

        # 4H Bestätigung
        ok, reason = check_multiframe(symbol, signal)
        if not ok:
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            return

        # ML-Filter
        if symbol in cfg.get("ml_symbols", []) and not self.ml_ok(symbol, last, signal):
            self.push_fn("skip", {"symbol": symbol, "signal": signal,
                                   "reason": "XGBoost lehnt Signal ab"})
            return

        # Trade ausführen
        self.last_signals[symbol] = signal
        self.execute_fn(
            symbol=symbol,
            signal=signal,
            rsi=last["rsi"],
            ema=last["ema_slow"],
            atr=last["atr"],
        )

    def run(self):
        """Haupt-Loop: scannt alle Symbole im Takt."""
        import screener as sc

        with open("config.json") as f:
            cfg = json.load(f)

        interval        = cfg.get("scan_interval_seconds", 300)
        screener_hour   = cfg.get("screener_run_hour", 9)  # Uhr NY-Zeit
        last_screen_day = None

        print(f"[Scanner] Gestartet | Interval: {interval}s")

        while True:
            self.last_scan = datetime.now().strftime("%H:%M:%S")

            # Marktzeiten prüfen
            ok, reason = broker.check_market_hours()
            if not ok:
                print(f"[Scanner] {reason} — warte...")
                self.push_fn("scanner", {"status": reason, "last_scan": self.last_scan})
                time.sleep(60)
                continue

            # Tägliches Verlustlimit
            if not broker.check_daily_loss_limit():
                print("[Scanner] Verlustlimit erreicht — kein Scan")
                time.sleep(60)
                continue

            with open("config.json") as f:
                cfg = json.load(f)

            # Screener einmal täglich morgens laufen lassen
            ny      = pytz.timezone("America/New_York")
            now_ny  = datetime.now(ny)
            today   = now_ny.strftime("%Y-%m-%d")
            if last_screen_day != today and now_ny.hour >= screener_hour:
                print("[Scanner] Starte täglichen Screener...")
                sc.run_screener(push_fn=self.push_fn, max_results=10)
                last_screen_day = today

            # Aktive Symbole: Basis + Screener-Kandidaten
            symbols = sc.get_active_symbols()
            symbols = list(dict.fromkeys(symbols))

            self.push_fn("scanner", {
                "status":    "Scanne Märkte...",
                "last_scan": self.last_scan,
                "symbols":   symbols,
            })

            for symbol in symbols:
                try:
                    self.scan_symbol(symbol, cfg)
                except Exception as e:
                    print(f"[Scanner] Fehler bei {symbol}: {e}")

            self.push_fn("scanner", {
                "status":    f"Warte {interval//60} Min bis zum nächsten Scan",
                "last_scan": self.last_scan,
            })

            time.sleep(interval)
