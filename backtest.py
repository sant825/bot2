"""
Backtesting-Modul: Testet die Trading-Strategie an historischen Daten.
Verwendet dieselbe Signal-Logik wie der Live-Scanner (Tages-Bars).

Aufruf (CLI):
  python backtest.py --symbols AAPL GLD SPY --days 365 --rr 2.0
"""

import argparse
import json
from datetime import datetime, timedelta
from typing import Optional
import pytz
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_historical_bars(symbol: str, days: int = 365) -> pd.DataFrame:
    """Holt historische Tagesbars von Alpaca IEX."""
    import broker
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    dc    = broker.get_data_client()
    end   = datetime.now(pytz.UTC)
    start = end - timedelta(days=days + 60)   # Puffer für Indikatoren
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
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
        "date":   str(b.timestamp)[:10],
        "open":   b.open,
        "high":   b.high,
        "low":    b.low,
        "close":  b.close,
        "volume": b.volume,
    } for b in bar_list]
    return pd.DataFrame(rows)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle Indikatoren — identisch zum Live-Scanner."""
    df = df.copy()

    # RSI (EWM wie im Scanner)
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # EMAs
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # ATR
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    # Volumen-Ratio
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df.dropna().reset_index(drop=True)


def run_backtest(symbol: str, days: int = 365,
                 rsi_oversold: int = 30, rsi_overbought: int = 70,
                 rr_ratio: float = 2.0, atr_multiplier: float = 2.5,
                 vol_threshold: float = 0.8, max_hold_days: int = 5) -> dict:
    """
    Führt Backtest für ein Symbol durch.
    - Entry: nächster Tag nach Signal (am Open)
    - Exit:  SL/TP via High/Low der Folgetage; Timeout nach max_hold_days
    Returns dict mit trades (letzte 50) + metrics + equity-Kurve.
    """
    print(f"[Backtest] {symbol}: Lade {days} Tage Daten...")
    df = get_historical_bars(symbol, days)
    if df.empty:
        return {"symbol": symbol, "error": "Keine Daten von Alpaca erhalten"}
    if len(df) < 220:
        return {"symbol": symbol, "error": f"Zu wenig Daten ({len(df)} Bars, mind. 220 benötigt)"}

    df = calculate_indicators(df)

    trades     = []
    in_position = False

    for i in range(1, len(df) - 2):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_position:
            continue   # Kein Pyramiding — warte auf Exit

        # Signal: RSI-Umkehr + EMA200-Trendbestätigung (Fix 2 gespiegelt)
        buy_sig  = (prev["rsi"] < rsi_oversold  and row["rsi"] >= rsi_oversold
                    and row["close"] > row["ema200"]   # Langfrist-Aufwärtstrend
                    and row["ema50"] > row["ema200"])   # EMA-Alignment bullisch
        sell_sig = (prev["rsi"] > rsi_overbought and row["rsi"] <= rsi_overbought
                    and row["close"] < row["ema200"]   # Langfrist-Abwärtstrend
                    and row["ema50"] < row["ema200"])   # EMA-Alignment bärisch

        if row["vol_ratio"] < vol_threshold:
            continue   # Volumen-Filter

        if buy_sig:
            signal = "buy"
        elif sell_sig:
            signal = "sell"
        else:
            continue

        # ── Entry ─────────────────────────────────────────────────
        entry_bar   = df.iloc[i + 1]
        entry_price = entry_bar["open"]
        if entry_price <= 0:
            continue

        atr         = max(row["atr"], entry_price * 0.005)  # mind. 0.5%
        sl_dist     = atr * atr_multiplier

        if signal == "buy":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + sl_dist * rr_ratio
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - sl_dist * rr_ratio

        trade = {
            "entry_date":  entry_bar["date"],
            "signal":      signal,
            "entry_price": round(entry_price, 4),
            "sl_price":    round(sl_price, 4),
            "tp_price":    round(tp_price, 4),
        }

        # ── Exit simulieren ───────────────────────────────────────
        exit_found = False
        for j in range(i + 2, min(i + 2 + max_hold_days, len(df))):
            bar = df.iloc[j]

            if signal == "buy":
                # Worst case: SL vor TP prüfen
                if bar["low"] <= sl_price:
                    trade.update({"exit_date": bar["date"], "exit_price": round(sl_price, 4),
                                  "exit_type": "stop_loss",
                                  "pnl_pct": round((sl_price - entry_price) / entry_price * 100, 2)})
                    exit_found = True; break
                if bar["high"] >= tp_price:
                    trade.update({"exit_date": bar["date"], "exit_price": round(tp_price, 4),
                                  "exit_type": "take_profit",
                                  "pnl_pct": round((tp_price - entry_price) / entry_price * 100, 2)})
                    exit_found = True; break
            else:
                if bar["high"] >= sl_price:
                    trade.update({"exit_date": bar["date"], "exit_price": round(sl_price, 4),
                                  "exit_type": "stop_loss",
                                  "pnl_pct": round((entry_price - sl_price) / entry_price * 100, 2)})
                    exit_found = True; break
                if bar["low"] <= tp_price:
                    trade.update({"exit_date": bar["date"], "exit_price": round(tp_price, 4),
                                  "exit_type": "take_profit",
                                  "pnl_pct": round((entry_price - tp_price) / entry_price * 100, 2)})
                    exit_found = True; break

        if not exit_found:
            # Zeitbasierter Exit am letzten verfügbaren Bar
            end_idx = min(i + 1 + max_hold_days, len(df) - 1)
            ep      = df.iloc[end_idx]["close"]
            pnl_pct = (ep - entry_price) / entry_price * 100 if signal == "buy" \
                      else (entry_price - ep) / entry_price * 100
            trade.update({"exit_date":  df.iloc[end_idx]["date"],
                           "exit_price": round(ep, 4),
                           "exit_type":  "timeout",
                           "pnl_pct":    round(pnl_pct, 2)})

        trades.append(trade)
        in_position = False   # Nächstes Signal sofort erlaubt (konservativ)

    # ── Metriken ──────────────────────────────────────────────────
    if not trades:
        return {"symbol": symbol, "days": days, "trades": [],
                "equity": [100.0], "metrics": {"total_trades": 0}}

    pnls   = [t["pnl_pct"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Equity-Kurve (prozentual, kumulativ, Startkapital 100)
    equity = [100.0]
    for p in pnls:
        equity.append(round(equity[-1] * (1 + p / 100), 2))

    # Max Drawdown
    peak, max_dd = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    # Sharpe (vereinfacht, annualisiert auf ~252 Handelstage)
    if len(pnls) > 1:
        mean_r = sum(pnls) / len(pnls)
        var_r  = sum((p - mean_r) ** 2 for p in pnls) / len(pnls)
        std_r  = var_r ** 0.5
        sharpe = round(mean_r / std_r * (252 ** 0.5) / 100, 2) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    metrics = {
        "total_trades":  len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "timeouts":      sum(1 for t in trades if t.get("exit_type") == "timeout"),
        "win_rate":      round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 0,
        "total_return":  round(equity[-1] - 100, 2),
        "avg_win":       round(sum(wins) / len(wins), 2)   if wins   else 0,
        "avg_loss":      round(sum(losses) / len(losses), 2) if losses else 0,
        "max_drawdown":  round(max_dd, 2),
        "sharpe_ratio":  sharpe,
        "best_trade":    round(max(pnls), 2),
        "worst_trade":   round(min(pnls), 2),
        "final_equity":  round(equity[-1], 2),
    }

    # Datumslabels für Equity-Kurve
    dates = [trades[0]["entry_date"]] + [t["exit_date"] for t in trades]

    return {
        "symbol":  symbol,
        "days":    days,
        "trades":  trades[-50:],   # Letzte 50 fürs Dashboard
        "equity":  equity,
        "dates":   dates,
        "metrics": metrics,
    }


def run_multi_backtest(symbols: list, days: int = 365, **kwargs) -> dict:
    """Backtest für mehrere Symbole, gibt Dict {symbol: result} zurück."""
    results = {}
    for sym in symbols:
        try:
            r = run_backtest(sym, days, **kwargs)
            m = r.get("metrics", {})
            results[sym] = r
            print(f"[Backtest] {sym}: {m.get('total_trades',0)} Trades | "
                  f"Win {m.get('win_rate',0)}% | PF {m.get('profit_factor',0)} | "
                  f"Return {m.get('total_return',0):+.1f}% | "
                  f"MaxDD -{m.get('max_drawdown',0):.1f}%")
        except Exception as e:
            print(f"[Backtest] Fehler {sym}: {e}")
            results[sym] = {"symbol": sym, "error": str(e)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot Backtester")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GLD", "SPY"],
                        help="Symbole (z.B. AAPL GLD SPY)")
    parser.add_argument("--days",    type=int,   default=365,  help="Historische Tage")
    parser.add_argument("--rr",      type=float, default=2.0,  help="Risk/Reward Ratio")
    parser.add_argument("--rsi-os",  type=int,   default=30,   help="RSI Oversold Level")
    parser.add_argument("--rsi-ob",  type=int,   default=70,   help="RSI Overbought Level")
    parser.add_argument("--vol",     type=float, default=0.8,  help="Volumen-Schwelle")
    args = parser.parse_args()

    results = run_multi_backtest(
        args.symbols, days=args.days, rr_ratio=args.rr,
        rsi_oversold=args.rsi_os, rsi_overbought=args.rsi_ob,
        vol_threshold=args.vol,
    )
    print("\n═══ ZUSAMMENFASSUNG ═══")
    print(json.dumps({
        sym: r.get("metrics", r.get("error", {}))
        for sym, r in results.items()
    }, indent=2, ensure_ascii=False))
