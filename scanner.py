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
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

import broker
import screener as sc

# Cache damit SPY nicht bei jedem Symbol neu geladen wird
_regime_cache = {"ts": None, "regime": None, "atr_pct": None}
_REGIME_TTL   = 300   # Sekunden bis Cache abläuft

# Earnings-Datum Cache {symbol: (fetched_ts, next_date_or_None)}
_earnings_cache: dict = {}
_EARNINGS_TTL = 3600 * 24  # 24 Stunden


def _get_next_earnings(symbol: str):
    """Gibt nächstes Earnings-Datum zurück oder None bei Fehler (nie blockierend)."""
    now = datetime.now(pytz.UTC).timestamp()
    cached = _earnings_cache.get(symbol)
    if cached and now - cached[0] < _EARNINGS_TTL:
        return cached[1]
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        cal = t.calendar
        earn = None
        if cal is not None:
            if hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                earn = cal["Earnings Date"].iloc[0]
            elif isinstance(cal, dict):
                earn = cal.get("Earnings Date") or cal.get("earningsDate")
                if isinstance(earn, list):
                    earn = earn[0] if earn else None
        if earn is not None and hasattr(earn, "to_pydatetime"):
            earn = earn.to_pydatetime()
        _earnings_cache[symbol] = (now, earn)
        return earn
    except Exception:
        _earnings_cache[symbol] = (now, None)
        return None


def check_earnings_safe(symbol: str, cfg: dict) -> tuple[bool, str]:
    """True = sicher zu handeln; False = zu nah an Earnings."""
    days_buf = cfg.get("earnings_days_buffer", 5)
    earn = _get_next_earnings(symbol)
    if earn is None:
        return True, ""
    now_utc = datetime.now(pytz.UTC)
    if not hasattr(earn, "tzinfo") or earn.tzinfo is None:
        earn = pytz.UTC.localize(earn)
    delta_days = (earn - now_utc).days
    if -2 <= delta_days <= days_buf:
        return False, f"Earnings in {delta_days}T ({earn.strftime('%d.%m.')}) — skip"
    return True, ""


def get_market_regime(cfg: dict) -> tuple[str, float]:
    """
    Prüft den Markt-Zustand anhand von SPY:
    - regime: 'bull' | 'bear' | 'neutral'
    - atr_pct: aktuelle Volatilität in % (ATR / Kurs)
    Ergebnis wird 5 Minuten gecacht.
    """
    global _regime_cache
    now = datetime.now(pytz.UTC).timestamp()
    if _regime_cache["ts"] and now - _regime_cache["ts"] < _REGIME_TTL:
        return _regime_cache["regime"], _regime_cache["atr_pct"]

    try:
        df = get_bars_df("SPY", TimeFrame.Hour, limit=220)
        if df.empty or len(df) < 55:
            return "neutral", 0.0

        df = calculate_indicators(df)
        if df.empty:
            return "neutral", 0.0

        last    = df.iloc[-1]
        close   = last["close"]
        ema50   = last["ema_fast"]
        ema200  = last["ema_slow"]
        atr_pct = round(last["atr"] / close * 100, 2) if close > 0 else 0.0

        bear_threshold = cfg.get("bear_filter_pct", 2.0)
        bull_threshold = cfg.get("bull_filter_pct", 1.0)
        ema50_dist_pct = (close - ema50) / ema50 * 100

        if close > ema50 and ema50 > ema200 and ema50_dist_pct >= bull_threshold:
            regime = "bull"
        elif close < ema50 and ema50 < ema200 and ema50_dist_pct <= -bear_threshold:
            regime = "bear"
        else:
            regime = "neutral"

        _regime_cache = {"ts": now, "regime": regime, "atr_pct": atr_pct}
        print(f"[Regime] SPY: {regime.upper()} | ATR%={atr_pct} | "
              f"Kurs={close:.2f} EMA50={ema50:.2f} EMA200={ema200:.2f} | "
              f"Dist={ema50_dist_pct:+.1f}%")
        return regime, atr_pct

    except Exception as e:
        print(f"[Regime] Fehler: {e}")
        return "neutral", 0.0


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
        feed=DataFeed.IEX,
    )
    bars = dc.get_stock_bars(req)
    try:
        bar_list = bars[symbol]
        if not bar_list:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    rows = [{
        "open":   b.open,
        "high":   b.high,
        "low":    b.low,
        "close":  b.close,
        "volume": b.volume,
    } for b in bar_list]
    return pd.DataFrame(rows)


def calculate_indicators(df: pd.DataFrame, rsi_period: int = 14,
                          ema_fast: int = 50, ema_slow: int = 200) -> pd.DataFrame:
    """Berechnet RSI, EMA50, EMA200, ATR, Volumen-Ratio."""
    df = df.copy()

    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).ewm(span=rsi_period, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(span=rsi_period, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    df["ema_fast"] = df["close"].ewm(span=ema_fast,  adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    df["tr"]  = (df["high"] - df["low"]).rolling(14).mean()
    df["atr"] = df["tr"]

    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df.dropna()


def check_signal(df: pd.DataFrame, cfg: dict) -> Optional[str]:
    """
    Prüft ob ein BUY oder SELL Signal vorliegt (für Basis-Symbole ohne Screener-Eintrag).
    """
    if len(df) < 4:
        return None

    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    rsi_os = cfg["rsi"]["oversold"]
    rsi_ob = cfg["rsi"]["overbought"]

    window = df.iloc[-15:]
    rsi_was_oversold   = (window["rsi"] < rsi_os).any()
    rsi_was_overbought = (window["rsi"] > rsi_ob).any()

    buy = (
        rsi_was_oversold
        and last["rsi"] >= rsi_os
        and last["rsi"] < 50
        and last["rsi"] > prev["rsi"]
        and last["close"] > last["ema_fast"]
    )

    sell = (
        rsi_was_overbought
        and last["rsi"] <= rsi_ob
        and last["rsi"] > 50
        and last["rsi"] < prev["rsi"]
        and last["close"] < last["ema_fast"]
    )

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


def check_15min_confirmation(symbol: str, signal: str) -> tuple[bool, str]:
    """15-Minuten-Bestätigung: Kurzfristiger RSI darf nicht bereits überhitzt sein."""
    try:
        tf_15m = TimeFrame(15, TimeFrameUnit.Minute)
        df = get_bars_df(symbol, tf_15m, limit=60)
        if df.empty or len(df) < 15:
            return True, ""

        df = calculate_indicators(df)
        if df.empty:
            return True, ""

        rsi = df.iloc[-1]["rsi"]
        if signal == "buy" and rsi > 68:
            return False, f"15min RSI={rsi:.1f} überhitzt — Entry zu spät"
        if signal == "sell" and rsi < 32:
            return False, f"15min RSI={rsi:.1f} überverkauft — Entry zu spät"
        return True, ""
    except Exception as e:
        print(f"[15min] Fehler {symbol}: {e}")
        return True, ""


class Scanner:
    def __init__(self, execute_fn, push_fn):
        self.execute_fn      = execute_fn
        self.push_fn         = push_fn
        self.models          = {}
        self.last_scan       = None
        self.last_regime     = None
        self._entry_times: dict  = {}   # {symbol: datetime} — Einstiegszeit
        self._partial_taken: dict = {}  # {symbol: bool} — Breakeven Stop gesetzt?

        for sym in ("AAPL", "GLD", "SPY"):
            m = xgb.XGBClassifier()
            try:
                m.load_model(f"model_{sym}.json")
                self.models[sym] = m
            except Exception:
                pass

    def _get_position_entry_time(self, symbol: str):
        """Holt Einstiegszeit einer Position aus Alpaca-Orders (lazy, einmalig)."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            client = broker.get_client()
            since  = datetime.now(pytz.UTC) - timedelta(days=7)
            req    = GetOrdersRequest(status=QueryOrderStatus.ALL, after=since, limit=50)
            orders = client.get_orders(req)
            for o in orders:
                if o.symbol == symbol and "filled" in str(o.status).lower():
                    if o.filled_at:
                        return o.filled_at
        except Exception as e:
            print(f"[EntryTime] Fehler {symbol}: {e}")
        return None

    def _resolve_entry_time(self, symbol: str):
        """Gibt gecachte Einstiegszeit zurück; holt sie bei Bedarf von Alpaca."""
        if symbol not in self._entry_times:
            t = self._get_position_entry_time(symbol)
            if t:
                self._entry_times[symbol] = t
        return self._entry_times.get(symbol)

    def check_rsi_exits(self, cfg: dict):
        """Schließt Positionen wenn RSI Extremwerte erreicht (Exit-Strategie)."""
        rsi_ob    = cfg["rsi"]["overbought"]
        rsi_os    = cfg["rsi"]["oversold"]
        min_hold  = cfg.get("min_hold_minutes", 30)
        now       = datetime.now(pytz.UTC)

        try:
            positions = broker.get_open_positions()
        except Exception as e:
            print(f"[Exit] Positionen laden fehlgeschlagen: {e}")
            return

        for p in positions:
            sym  = p.symbol
            side = str(p.side.value)

            # Fix 3: Mindest-Haltezeit — nicht sofort nach Einstieg schließen
            entry_time = self._resolve_entry_time(sym)
            if entry_time:
                age_min = (now - entry_time).total_seconds() / 60
                if age_min < min_hold:
                    print(f"[Exit] {sym}: {age_min:.0f} Min jung — warte {min_hold} Min Mindesthaltezeit")
                    continue

            try:
                df = get_bars_df(sym, TimeFrame.Hour, limit=50)
                if df.empty:
                    continue
                df  = calculate_indicators(df)
                if df.empty:
                    continue
                rsi = df.iloc[-1]["rsi"]

                if side == "long" and rsi > rsi_ob:
                    print(f"[Exit] {sym} Long: RSI={rsi:.1f} > {rsi_ob} → schließe")
                    broker.close_position(sym)
                    self._entry_times.pop(sym, None)
                    self._partial_taken.pop(sym, None)
                    self.push_fn("skip", {
                        "symbol": sym, "signal": "exit",
                        "reason": f"RSI-Exit: {rsi:.1f} überkauft — Long geschlossen",
                    })
                elif side == "short" and rsi < rsi_os:
                    print(f"[Exit] {sym} Short: RSI={rsi:.1f} < {rsi_os} → schließe")
                    broker.close_position(sym)
                    self._entry_times.pop(sym, None)
                    self._partial_taken.pop(sym, None)
                    self.push_fn("skip", {
                        "symbol": sym, "signal": "exit",
                        "reason": f"RSI-Exit: {rsi:.1f} überverkauft — Short geschlossen",
                    })
            except Exception as e:
                print(f"[Exit] Fehler {sym}: {e}")

    def update_trailing_stops(self, cfg: dict):
        """
        Zieht Stop-Loss für profitable Positionen nach.
        Fix 7: Setzt Stop auf Breakeven wenn 1:1 R:R erreicht ist.
        Fix 3: Wartet Mindesthaltezeit ab bevor Stops nachgezogen werden.
        """
        trail_pct = cfg["risk_management"].get("trailing_stop_percent", 2.5) / 100
        min_hold  = cfg.get("min_hold_minutes", 30)
        now       = datetime.now(pytz.UTC)

        try:
            from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
            from alpaca.trading.enums import QueryOrderStatus

            client    = broker.get_client()
            positions = broker.get_open_positions()

            req         = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
            open_orders = client.get_orders(req)

            stop_orders = {}
            for o in open_orders:
                if "stop" in str(o.order_type).lower():
                    stop_orders[o.symbol] = o

            for p in positions:
                sym  = p.symbol
                side = str(p.side.value)

                # Fix 3: Mindest-Haltezeit
                entry_time = self._resolve_entry_time(sym)
                if entry_time:
                    age_min = (now - entry_time).total_seconds() / 60
                    if age_min < min_hold:
                        continue

                entry   = float(p.avg_entry_price)
                current = float(p.current_price)

                stop_order = stop_orders.get(sym)
                if not stop_order:
                    continue

                current_stop = float(stop_order.stop_price or 0)
                if current_stop == 0:
                    continue

                new_stop = current_stop  # Startwert: aktueller Stop

                # Fix 7: Breakeven Stop bei 1:1 R:R
                if not self._partial_taken.get(sym):
                    sl_distance = abs(entry - current_stop)
                    if side == "long":
                        one_to_one = entry + sl_distance
                        if current >= one_to_one:
                            breakeven = round(entry * 1.001, 2)  # Entry + 0.1% Puffer
                            new_stop  = max(new_stop, breakeven)
                            self._partial_taken[sym] = True
                            print(f"[1:1] {sym}: 1:1 erreicht — Stop → Breakeven {breakeven:.2f} "
                                  f"(Entry {entry:.2f}, 1:1 bei {one_to_one:.2f})")
                    elif side == "short":
                        one_to_one = entry - sl_distance
                        if current <= one_to_one:
                            breakeven = round(entry * 0.999, 2)  # Entry - 0.1% Puffer
                            new_stop  = min(new_stop, breakeven) if new_stop > 0 else breakeven
                            self._partial_taken[sym] = True
                            print(f"[1:1] {sym}: 1:1 erreicht — Stop → Breakeven {breakeven:.2f} "
                                  f"(Entry {entry:.2f}, 1:1 bei {one_to_one:.2f})")

                # Trailing Stop (nur wenn > 1% im Plus)
                pnl_pct = (current - entry) / entry if side == "long" else (entry - current) / entry
                if pnl_pct >= 0.01:
                    if side == "long":
                        trail_stop = round(current * (1 - trail_pct), 2)
                        new_stop   = max(new_stop, trail_stop)
                    elif side == "short":
                        trail_stop = round(current * (1 + trail_pct), 2)
                        if new_stop == current_stop:
                            new_stop = trail_stop
                        else:
                            new_stop = min(new_stop, trail_stop)

                if new_stop == current_stop:
                    continue

                req_replace = ReplaceOrderRequest(stop_price=new_stop)
                client.replace_order_by_id(stop_order.id, req_replace)
                print(f"[Trail] {sym}: Stop {current_stop:.2f} → {new_stop:.2f} "
                      f"(Kurs {current:.2f}, {pnl_pct*100:+.1f}%)")

        except Exception as e:
            print(f"[Trail] Fehler: {e}")

    def check_stale_positions(self, cfg: dict):
        """
        Fix 5: Schließt Positionen die älter als max_hold_days sind.
        Verhindert dass schlechte Trades tagelang Kapital binden.
        """
        max_hold_days = cfg.get("max_hold_days", 2)
        now = datetime.now(pytz.UTC)

        try:
            positions = broker.get_open_positions()
        except Exception as e:
            print(f"[Stale] Fehler beim Laden: {e}")
            return

        for p in positions:
            sym = p.symbol
            pnl = float(p.unrealized_pl or 0)

            entry_time = self._resolve_entry_time(sym)
            if entry_time is None:
                continue

            age_hours = (now - entry_time).total_seconds() / 3600
            if age_hours < max_hold_days * 24:
                continue

            print(f"[Stale] {sym}: {age_hours:.0f}h alt (>{max_hold_days}d) | "
                  f"PnL: {pnl:+.2f}$ → schließe")
            try:
                broker.close_position(sym)
                self._entry_times.pop(sym, None)
                self._partial_taken.pop(sym, None)
                self.push_fn("skip", {
                    "symbol": sym,
                    "signal": "exit",
                    "reason": f"Zeitbasierter Exit nach {age_hours:.0f}h — Position geschlossen",
                })
            except Exception as e:
                print(f"[Stale] Fehler beim Schließen {sym}: {e}")

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

        # ── 1. Screener-Score-Filter ─────────────────────────────
        min_score = cfg.get("min_screener_score", 65)
        score = None
        screener_entry = next((r for r in sc.screener_results if r["symbol"] == symbol), None)

        if symbol not in sc.BASE_SYMBOLS:
            if screener_entry is None:
                print(f"[Scanner] {symbol}: Nicht im Screener — überspringe")
                return
            if screener_entry["score"] < min_score:
                print(f"[Scanner] {symbol}: Score {screener_entry['score']} < {min_score} — überspringe")
                return
            score = screener_entry["score"]
        else:
            if screener_entry:
                score = screener_entry["score"]

        # Stunden-Bars für Timing und Indikatoren
        df = get_bars_df(symbol, TimeFrame.Hour, limit=250)
        if df.empty:
            print(f"[Scanner] Keine Daten für {symbol}")
            return

        df = calculate_indicators(df)
        if df.empty:
            return

        last = df.iloc[-1]

        # Signalquelle: Screener-Signal (Tages-RSI) wenn vorhanden
        if screener_entry:
            signal     = screener_entry["signal"]
            hourly_rsi = last["rsi"]
            if signal == "buy" and hourly_rsi > 68:
                print(f"[Scanner] {symbol}: Screener BUY aber 1H RSI={hourly_rsi:.1f} überhitzt — warte")
                return
            if signal == "sell" and hourly_rsi < 32:
                print(f"[Scanner] {symbol}: Screener SELL aber 1H RSI={hourly_rsi:.1f} überverkauft — warte")
                return
        else:
            signal = check_signal(df, cfg)
            if not signal:
                return

        print(f"[Scanner] Signal: {signal.upper()} {symbol} | 1H RSI={last['rsi']:.1f}"
              + (f" | Score={score}" if score else ""))

        # ── 2. Offene Position prüfen ────────────────────────────
        try:
            open_positions = broker.get_open_positions()
            open_map = {str(p.symbol): str(p.side.value) for p in open_positions}
        except Exception:
            open_map = {}

        current_side = open_map.get(symbol)
        if (signal == "buy"  and current_side == "long") or \
           (signal == "sell" and current_side == "short"):
            print(f"[Scanner] {symbol}: Position bereits offen ({current_side}) — überspringe")
            return

        # ── 3. Volumen prüfen ─────────────────────────────────────
        vol_threshold = cfg.get("volume_threshold", 0.8)
        if last["vol_ratio"] < vol_threshold:
            reason = f"Volumen zu niedrig ({last['vol_ratio']:.2f}x Durchschnitt)"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            return

        # ── 4. Earnings-Filter (Fix 1) ────────────────────────────
        ok, reason = check_earnings_safe(symbol, cfg)
        if not ok:
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[Earnings] Skip {symbol}: {reason}")
            return

        # ── 5. Individuelle EMA200-Ausrichtung (Fix 2) ────────────
        # Screener berechnet EMA200 auf Tagesbars — zuverlässiger als stündliche EMA200
        if screener_entry:
            above_ema200 = screener_entry.get("above_ema200", None)
            if above_ema200 is not None:
                if signal == "buy" and not above_ema200:
                    reason = f"Tages-EMA200: {symbol} im Abwärtstrend — kein BUY"
                    self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
                    print(f"[EMA200] Skip {symbol}: {reason}")
                    return
                if signal == "sell" and above_ema200:
                    reason = f"Tages-EMA200: {symbol} im Aufwärtstrend — kein SELL"
                    self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
                    print(f"[EMA200] Skip {symbol}: {reason}")
                    return

        # ── 6. 4H Bestätigung ─────────────────────────────────────
        ok, reason = check_multiframe(symbol, signal)
        if not ok:
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            return

        # ── 7. 15-Minuten-Bestätigung ─────────────────────────────
        ok, reason = check_15min_confirmation(symbol, signal)
        if not ok:
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[15min] Skip {symbol}: {reason}")
            return

        # ── 8. Market-Regime-Filter ──────────────────────────────
        regime, atr_pct = get_market_regime(cfg)
        max_atr     = cfg.get("max_spy_atr_pct", 2.5)
        skip_vol    = cfg.get("skip_on_extreme_volatility", True)
        safe_havens = cfg.get("safe_haven_symbols", ["GLD", "GDX", "TLT", "SLV"])

        if skip_vol and atr_pct > max_atr:
            reason = f"Extreme Marktvolatilität (SPY ATR {atr_pct}% > {max_atr}%)"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[Regime] Skip {symbol}: {reason}")
            return

        is_safe_haven = symbol in safe_havens

        # Kein BUY außerhalb klarem Bull-Regime (Fix 3 — auch neutral blockiert)
        if regime in ("bear", "neutral") and signal == "buy" and not is_safe_haven:
            reason = f"Regime {regime.upper()}: SPY kein klarer Aufwärtstrend — kein BUY"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[Regime] Skip {symbol}: {reason}")
            return

        # Kein SELL außerhalb klarem Bear-Regime
        if regime in ("bull", "neutral") and signal == "sell" and not is_safe_haven:
            reason = f"Regime {regime.upper()}: SPY kein klarer Abwärtstrend — kein SELL"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[Regime] Skip {symbol}: {reason}")
            return

        # ── 9. ML-Filter ─────────────────────────────────────────
        if symbol in cfg.get("ml_symbols", []) and not self.ml_ok(symbol, last, signal):
            self.push_fn("skip", {"symbol": symbol, "signal": signal,
                                   "reason": "XGBoost lehnt Signal ab"})
            return

        # ── 10. Trade ausführen ───────────────────────────────────
        self.execute_fn(
            symbol=symbol,
            signal=signal,
            rsi=last["rsi"],
            ema=last["ema_slow"],
            atr=last["atr"],
            score=score,
        )

        # Fix 3: Einstiegszeit merken für Mindesthaltezeit + zeitbasierten Exit
        self._entry_times[symbol] = datetime.now(pytz.UTC)
        self._partial_taken.pop(symbol, None)  # Reset für neuen Trade

    def run(self):
        """Haupt-Loop: scannt alle Symbole im Takt."""
        with open("config.json") as f:
            cfg = json.load(f)

        interval          = cfg.get("scan_interval_seconds", 300)
        screener_hour     = cfg.get("screener_run_hour", 9)
        last_screen_day   = None
        last_retrain_week = None

        print(f"[Scanner] Gestartet | Interval: {interval}s")

        while True:
            self.last_scan = datetime.now().strftime("%H:%M:%S")

            with open("config.json") as f:
                cfg = json.load(f)

            ny     = pytz.timezone("America/New_York")
            now_ny = datetime.now(ny)
            today  = now_ny.strftime("%Y-%m-%d")

            # Screener einmal täglich morgens
            if last_screen_day != today and now_ny.hour >= screener_hour:
                print("[Scanner] Starte täglichen Screener...")
                sc.run_screener(push_fn=self.push_fn, max_results=10)
                last_screen_day = today

            # Wöchentliches ML-Retraining (Sonntag)
            week_num = now_ny.isocalendar()[1]
            if now_ny.weekday() == 6 and last_retrain_week != week_num:
                print("[Scanner] Starte wöchentliches ML-Retraining...")
                try:
                    import auto_retrain
                    auto_retrain.retrain_all(push_fn=self.push_fn)
                    last_retrain_week = week_num
                    for sym in ("AAPL", "GLD", "SPY"):
                        m = xgb.XGBClassifier()
                        try:
                            m.load_model(f"model_{sym}.json")
                            self.models[sym] = m
                            print(f"[AutoRetrain] Modell neu geladen: {sym}")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[AutoRetrain] Fehler: {e}")

            symbols = sc.get_active_symbols()
            symbols = list(dict.fromkeys(symbols))

            # Marktzeiten prüfen
            ok, reason = broker.check_market_hours()
            if not ok:
                print(f"[Scanner] {reason} — warte...")
                self.push_fn("scanner", {
                    "status":    reason,
                    "last_scan": self.last_scan,
                    "symbols":   symbols,
                })
                time.sleep(60)
                continue

            # Tägliches Verlustlimit
            if not broker.check_daily_loss_limit():
                print("[Scanner] Verlustlimit erreicht — kein Scan")
                self.push_fn("scanner", {
                    "status":    "Verlustlimit erreicht",
                    "last_scan": self.last_scan,
                    "symbols":   symbols,
                })
                time.sleep(60)
                continue

            regime, atr_pct = get_market_regime(cfg)

            # Regime-Wechsel: offene Positionen schließen
            if self.last_regime and self.last_regime != regime and regime != "neutral":
                positions = broker.get_open_positions()
                for p in positions:
                    side = str(p.side.value)
                    if regime == "bear" and side == "long":
                        print(f"[Regime] Wechsel zu BEAR → schließe Long {p.symbol}")
                        try:
                            broker.close_position(p.symbol)
                            self._entry_times.pop(p.symbol, None)
                            self._partial_taken.pop(p.symbol, None)
                            self.push_fn("skip", {
                                "symbol": p.symbol, "signal": "exit",
                                "reason": f"Regime-Wechsel zu Bärenmarkt — Long {p.symbol} geschlossen",
                            })
                        except Exception as e:
                            print(f"[Regime] Fehler beim Schließen {p.symbol}: {e}")
                    elif regime == "bull" and side == "short":
                        print(f"[Regime] Wechsel zu BULL → schließe Short {p.symbol}")
                        try:
                            broker.close_position(p.symbol)
                            self._entry_times.pop(p.symbol, None)
                            self._partial_taken.pop(p.symbol, None)
                            self.push_fn("skip", {
                                "symbol": p.symbol, "signal": "exit",
                                "reason": f"Regime-Wechsel zu Bullenmarkt — Short {p.symbol} geschlossen",
                            })
                        except Exception as e:
                            print(f"[Regime] Fehler beim Schließen {p.symbol}: {e}")

            self.last_regime = regime

            self.push_fn("scanner", {
                "status":    "Scanne Märkte...",
                "last_scan": self.last_scan,
                "symbols":   symbols,
                "regime":    regime,
                "atr_pct":   atr_pct,
            })

            # Fix 5: Zeitbasierter Exit — abgelaufene Positionen schließen
            try:
                self.check_stale_positions(cfg)
            except Exception as e:
                print(f"[Stale] Fehler: {e}")

            # RSI-Exit Check
            try:
                self.check_rsi_exits(cfg)
            except Exception as e:
                print(f"[Exit] Fehler: {e}")

            # Trailing Stops nachziehen (inkl. Breakeven)
            try:
                self.update_trailing_stops(cfg)
            except Exception as e:
                print(f"[Trail] Fehler: {e}")

            for symbol in symbols:
                try:
                    self.scan_symbol(symbol, cfg)
                except Exception as e:
                    print(f"[Scanner] Fehler bei {symbol}: {e}")

            self.push_fn("scanner", {
                "status":    f"Warte {interval//60} Min bis zum nächsten Scan",
                "last_scan": self.last_scan,
                "symbols":   symbols,
            })

            time.sleep(interval)
