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
        ema50   = last["ema_fast"]   # ema_fast = EMA50
        ema200  = last["ema_slow"]   # ema_slow = EMA200
        atr_pct = round(last["atr"] / close * 100, 2) if close > 0 else 0.0

        # Prozentuale Abweichung von EMA50 — verhindert dass 0.1% Abstand = BEAR
        bear_threshold = cfg.get("bear_filter_pct", 2.0)   # Standard: erst ab -2%
        bull_threshold = cfg.get("bull_filter_pct", 1.0)   # Standard: erst ab +1%
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

    Logik: RSI hat in den letzten 10 Stunden den Oversold/Overbought-Level
    gekreuzt UND steigt/fällt aktuell noch in die richtige Richtung.
    Das 10-Stunden-Fenster verhindert, dass ein Signal nach wenigen Minuten
    verfällt wenn der Scanner den exakten Kreuzungszeitpunkt knapp verpasst.
    """
    if len(df) < 4:
        return None

    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    rsi_os = cfg["rsi"]["oversold"]   # Standard: 30
    rsi_ob = cfg["rsi"]["overbought"] # Standard: 70

    # 15-Bar-Fenster (15 Stunden bei 1H-Bars) auf Kreuzung prüfen
    window = df.iloc[-15:]
    rsi_was_oversold  = (window["rsi"] < rsi_os).any()
    rsi_was_overbought = (window["rsi"] > rsi_ob).any()

    # BUY: RSI war kürzlich überverkauft, hat sich erholt, steigt noch und
    #       Kurs über EMA50 (mittelfristiger Aufwärtstrend)
    buy = (
        rsi_was_oversold
        and last["rsi"] >= rsi_os         # Über dem Oversold-Level
        and last["rsi"] < 50              # Noch Aufwärtspotenzial vorhanden
        and last["rsi"] > prev["rsi"]     # RSI steigt
        and last["close"] > last["ema_fast"]  # Über EMA50
    )

    # SELL: RSI war kürzlich überkauft, ist gefallen, fällt noch und
    #        Kurs unter EMA50 (mittelfristiger Abwärtstrend)
    sell = (
        rsi_was_overbought
        and last["rsi"] <= rsi_ob         # Unter dem Overbought-Level
        and last["rsi"] > 50              # Noch Abwärtspotenzial vorhanden
        and last["rsi"] < prev["rsi"]     # RSI fällt
        and last["close"] < last["ema_fast"]  # Unter EMA50
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
        """
        execute_fn: Funktion die einen Trade ausführt
        push_fn:    Funktion die SSE-Events ans Dashboard schickt
        """
        self.execute_fn   = execute_fn
        self.push_fn      = push_fn
        self.models       = {}
        self.last_scan    = None
        self.last_regime  = None   # Für Regime-Wechsel-Erkennung

        # ML-Modelle laden
        for sym in ("AAPL", "GLD", "SPY"):
            m = xgb.XGBClassifier()
            try:
                m.load_model(f"model_{sym}.json")
                self.models[sym] = m
            except Exception:
                pass

    def check_rsi_exits(self, cfg: dict):
        """Schließt Positionen wenn RSI Extremwerte erreicht (Exit-Strategie)."""
        rsi_ob = cfg["rsi"]["overbought"]
        rsi_os = cfg["rsi"]["oversold"]

        try:
            positions = broker.get_open_positions()
        except Exception as e:
            print(f"[Exit] Positionen laden fehlgeschlagen: {e}")
            return

        for p in positions:
            sym  = p.symbol
            side = str(p.side.value)
            try:
                df = get_bars_df(sym, TimeFrame.Hour, limit=50)
                if df.empty:
                    continue
                df = calculate_indicators(df)
                if df.empty:
                    continue
                rsi = df.iloc[-1]["rsi"]

                if side == "long" and rsi > rsi_ob:
                    print(f"[Exit] {sym} Long: RSI={rsi:.1f} > {rsi_ob} → schließe")
                    broker.close_position(sym)
                    self.push_fn("skip", {
                        "symbol": sym, "signal": "exit",
                        "reason": f"RSI-Exit: {rsi:.1f} überkauft — Long geschlossen",
                    })
                elif side == "short" and rsi < rsi_os:
                    print(f"[Exit] {sym} Short: RSI={rsi:.1f} < {rsi_os} → schließe")
                    broker.close_position(sym)
                    self.push_fn("skip", {
                        "symbol": sym, "signal": "exit",
                        "reason": f"RSI-Exit: {rsi:.1f} überverkauft — Short geschlossen",
                    })
            except Exception as e:
                print(f"[Exit] Fehler {sym}: {e}")

    def update_trailing_stops(self, cfg: dict):
        """Zieht Stop-Loss für profitable Long-Positionen nach oben."""
        trail_pct = cfg["risk_management"].get("trailing_stop_percent", 1.5) / 100
        try:
            from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
            from alpaca.trading.enums import QueryOrderStatus

            client    = broker.get_client()
            positions = broker.get_open_positions()

            req         = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
            open_orders = client.get_orders(req)

            # Stop-Orders pro Symbol mit Order-ID
            stop_orders = {}
            for o in open_orders:
                if "stop" in str(o.order_type).lower():
                    stop_orders[o.symbol] = o

            for p in positions:
                sym  = p.symbol
                side = str(p.side.value)
                if side != "long":
                    continue

                entry   = float(p.avg_entry_price)
                current = float(p.current_price)
                pnl_pct = (current - entry) / entry

                if pnl_pct < 0.01:   # Nur wenn > 1% im Plus
                    continue

                new_stop   = round(current * (1 - trail_pct), 2)
                stop_order = stop_orders.get(sym)
                if not stop_order:
                    continue

                current_stop = float(stop_order.stop_price or 0)
                if new_stop <= current_stop:
                    continue   # Bereits auf diesem Niveau oder höher

                req_replace = ReplaceOrderRequest(stop_price=new_stop)
                client.replace_order_by_id(stop_order.id, req_replace)
                print(f"[Trail] {sym}: Stop {current_stop:.2f} → {new_stop:.2f} "
                      f"(Kurs {current:.2f}, +{pnl_pct*100:.1f}%)")
        except Exception as e:
            print(f"[Trail] Fehler: {e}")

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
        min_score = cfg.get("min_screener_score", 60)
        score = None
        if symbol not in sc.BASE_SYMBOLS:
            score_entry = next((r for r in sc.screener_results if r["symbol"] == symbol), None)
            if score_entry is None:
                print(f"[Scanner] {symbol}: Nicht im Screener — überspringe")
                return
            if score_entry["score"] < min_score:
                print(f"[Scanner] {symbol}: Score {score_entry['score']} < {min_score} — überspringe")
                return
            score = score_entry["score"]
        else:
            # Basis-Symbole: Score aus Screener wenn vorhanden
            score_entry = next((r for r in sc.screener_results if r["symbol"] == symbol), None)
            if score_entry:
                score = score_entry["score"]

        # Stunden-Bars für Timing und Indikatoren
        df = get_bars_df(symbol, TimeFrame.Hour, limit=250)
        if df.empty:
            print(f"[Scanner] Keine Daten für {symbol}")
            return

        df = calculate_indicators(df)
        if df.empty:
            return

        last = df.iloc[-1]

        # Signalquelle: Screener-Signal (Tages-RSI) wenn vorhanden,
        # sonst check_signal auf Stunden-Basis (für Basis-Symbole ohne Screener-Eintrag)
        screener_entry = next((r for r in sc.screener_results if r["symbol"] == symbol), None)
        if screener_entry:
            signal = screener_entry["signal"]   # "buy" oder "sell" vom Screener (Tages-RSI)
            hourly_rsi = last["rsi"]
            # Timing-Filter: nicht einsteigen wenn stündlicher RSI bereits zu weit gelaufen
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

        # ── 2. Offene Position prüfen (statt last_signals) ───────
        try:
            open_positions = broker.get_open_positions()
            open_map = {str(p.symbol): str(p.side.value) for p in open_positions}
        except Exception:
            open_map = {}

        current_side = open_map.get(symbol)
        if (signal == "buy" and current_side == "long") or \
           (signal == "sell" and current_side == "short"):
            print(f"[Scanner] {symbol}: Position bereits offen ({current_side}) — überspringe")
            return

        # ── 3. Volumen prüfen ─────────────────────────────────────
        vol_threshold = cfg.get("volume_threshold", 0.8)
        if last["vol_ratio"] < vol_threshold:
            reason = f"Volumen zu niedrig ({last['vol_ratio']:.2f}x Durchschnitt)"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            return

        # ── 4. 4H Bestätigung ─────────────────────────────────────
        ok, reason = check_multiframe(symbol, signal)
        if not ok:
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            return

        # ── 5. 15-Minuten-Bestätigung ─────────────────────────────
        ok, reason = check_15min_confirmation(symbol, signal)
        if not ok:
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[15min] Skip {symbol}: {reason}")
            return

        # ── 6. Market-Regime-Filter ──────────────────────────────
        regime, atr_pct = get_market_regime(cfg)
        max_atr      = cfg.get("max_spy_atr_pct", 2.5)
        skip_vol     = cfg.get("skip_on_extreme_volatility", True)
        safe_havens  = cfg.get("safe_haven_symbols", ["GLD", "GDX", "TLT", "SLV"])

        if skip_vol and atr_pct > max_atr:
            reason = f"Extreme Marktvolatilität (SPY ATR {atr_pct}% > {max_atr}%)"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[Regime] Skip {symbol}: {reason}")
            return

        # Safe-Haven Symbole (z.B. GLD) immer erlaubt — steigen oft in Bärenmärkten
        is_safe_haven = symbol in safe_havens

        # Kein BUY im Bären- oder neutralen Markt (Trend muss bestätigt sein)
        if regime in ("bear", "neutral") and signal == "buy" and not is_safe_haven:
            reason = f"Regime {regime.upper()}: SPY kein klarer Aufwärtstrend — kein BUY"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[Regime] Skip {symbol}: {reason}")
            return

        # Kein SELL im Bullen- oder neutralen Markt (Trend muss bestätigt sein)
        if regime in ("bull", "neutral") and signal == "sell" and not is_safe_haven:
            reason = f"Regime {regime.upper()}: SPY kein klarer Abwärtstrend — kein SELL"
            self.push_fn("skip", {"symbol": symbol, "signal": signal, "reason": reason})
            print(f"[Regime] Skip {symbol}: {reason}")
            return

        # ── 7. ML-Filter ─────────────────────────────────────────
        if symbol in cfg.get("ml_symbols", []) and not self.ml_ok(symbol, last, signal):
            self.push_fn("skip", {"symbol": symbol, "signal": signal,
                                   "reason": "XGBoost lehnt Signal ab"})
            return

        # ── 8. Trade ausführen ────────────────────────────────────
        self.execute_fn(
            symbol=symbol,
            signal=signal,
            rsi=last["rsi"],
            ema=last["ema_slow"],
            atr=last["atr"],
            score=score,
        )

    def run(self):
        """Haupt-Loop: scannt alle Symbole im Takt."""
        with open("config.json") as f:
            cfg = json.load(f)

        interval         = cfg.get("scan_interval_seconds", 300)
        screener_hour    = cfg.get("screener_run_hour", 9)  # Uhr NY-Zeit
        last_screen_day  = None
        last_retrain_week = None

        print(f"[Scanner] Gestartet | Interval: {interval}s")

        while True:
            self.last_scan = datetime.now().strftime("%H:%M:%S")

            with open("config.json") as f:
                cfg = json.load(f)

            ny      = pytz.timezone("America/New_York")
            now_ny  = datetime.now(ny)
            today   = now_ny.strftime("%Y-%m-%d")

            # Screener einmal täglich morgens laufen lassen (unabhängig von Marktzeiten)
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
                    # Neu trainierte Modelle laden
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

            # Aktive Symbole immer ans Dashboard senden
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

            # ── Regime-Wechsel: offene Positionen schließen ───────
            if self.last_regime and self.last_regime != regime and regime != "neutral":
                positions = broker.get_open_positions()
                for p in positions:
                    side = str(p.side.value)
                    # Regime wechselt zu Bär → Longs schließen
                    if regime == "bear" and side == "long":
                        print(f"[Regime] Wechsel zu BEAR → schließe Long {p.symbol}")
                        try:
                            broker.close_position(p.symbol)
                            self.push_fn("skip", {
                                "symbol": p.symbol,
                                "signal": "exit",
                                "reason": f"Regime-Wechsel zu Bärenmarkt — Long {p.symbol} geschlossen",
                            })
                        except Exception as e:
                            print(f"[Regime] Fehler beim Schließen {p.symbol}: {e}")
                    # Regime wechselt zu Bull → Shorts schließen
                    elif regime == "bull" and side == "short":
                        print(f"[Regime] Wechsel zu BULL → schließe Short {p.symbol}")
                        try:
                            broker.close_position(p.symbol)
                            self.push_fn("skip", {
                                "symbol": p.symbol,
                                "signal": "exit",
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

            # Exit-Check: offene Positionen bei RSI-Extremen schließen
            try:
                self.check_rsi_exits(cfg)
            except Exception as e:
                print(f"[Exit] Fehler: {e}")

            # Trailing Stops nachziehen
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
