"""
TradingView RSI Bot - Webhook Server mit Live Dashboard
Empfängt TradingView Alerts → XGBoost filtert Signal → Alpaca führt Order aus → Google Sheets loggt

Webhook URL: http://DEINE_IP:5000/webhook
Dashboard:   http://DEINE_IP:5000
"""

import os
import json
import queue
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv
import xgboost as xgb
import pandas as pd

import broker
import risk_manager
import sheets_logger

load_dotenv()

app = Flask(__name__)
trades_today = []
skipped_today = 0


def load_todays_trades():
    """Lädt beim Start die heutigen Orders von Alpaca (nach Neustart)."""
    global trades_today
    try:
        client = broker.get_client()
        today = datetime.now().strftime("%Y-%m-%d")
        orders = client.get_orders()
        count = 0
        for o in orders:
            order_date = str(o.created_at)[:10]
            if order_date == today and str(o.status) not in ("canceled", "expired", "rejected"):
                trades_today.append({
                    "symbol":      str(o.symbol),
                    "signal":      str(o.side.value),
                    "order_id":    str(o.id),
                    "quantity":    float(o.qty or 0),
                    "entry_price": float(o.filled_avg_price or 0),
                    "status":      str(o.status),
                })
                count += 1
        if count:
            print(f"Startup: {count} heutige Order(s) von Alpaca geladen.")
    except Exception as e:
        print(f"Startup: Konnte heutige Orders nicht laden: {e}")

# SSE Event Queue für alle verbundenen Browser
sse_clients = []
sse_lock = threading.Lock()


def push_event(event_type: str, data: dict):
    """Schickt ein SSE-Event an alle verbundenen Browser."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        for q in sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


# XGBoost Modell laden
model = xgb.XGBClassifier()
model_loaded = False
for fname in ("trading_model.json", "model.json"):
    try:
        model.load_model(fname)
        print(f"XGBoost Modell geladen: {fname}")
        model_loaded = True
        break
    except Exception:
        pass

if not model_loaded:
    print("WARNUNG: Kein XGBoost Modell gefunden. Nur RSI-Signal wird genutzt.")


def verify_secret(data: dict) -> bool:
    expected = os.getenv("WEBHOOK_SECRET", "")
    return data.get("secret", "") == expected


def ml_confirms_signal(rsi: float, entry_price: float, ema: float, atr: float, signal: str) -> bool:
    if not model_loaded:
        return True
    ema200_dist = (entry_price - ema) / ema * 100 if ema > 0 else 0
    ema50_dist  = ema200_dist * 0.5  # Näherung wenn kein EMA50 vorhanden
    df_input = pd.DataFrame(
        [[rsi, ema200_dist, ema50_dist, atr]],
        columns=["RSI", "EMA200_dist", "EMA50_dist", "ATR"]
    )
    prediction = model.predict(df_input)[0]
    if signal == "buy" and prediction == 1:
        print(f"XGBoost bestätigt BUY | RSI={rsi:.1f}")
        return True
    elif signal == "sell" and prediction == 0:
        print(f"XGBoost bestätigt SELL | RSI={rsi:.1f}")
        return True
    else:
        print(f"XGBoost ABLEHNT '{signal}' | RSI={rsi:.1f}")
        return False


def get_status_data() -> dict:
    try:
        value = broker.get_account_value()
        positions = broker.get_open_positions()
        client = broker.get_client()
        account = client.get_account()
        daily_pnl = float(account.equity) - float(account.last_equity)

        positions_list = []
        for p in positions:
            pnl = float(p.unrealized_pl or 0)
            positions_list.append({
                "symbol":   p.symbol,
                "side":     str(p.side.value),
                "qty":      float(p.qty),
                "entry":    round(float(p.avg_entry_price), 2),
                "current":  round(float(p.current_price), 2),
                "pnl":      round(pnl, 2),
                "pnl_pct":  round(float(p.unrealized_plpc or 0) * 100, 2),
            })

        return {
            "account_value":   f"{value:,.2f} $",
            "open_positions":  len(positions),
            "positions_list":  positions_list,
            "trades_today":    len(trades_today),
            "skipped_today":   skipped_today,
            "daily_pnl":       f"{daily_pnl:+.2f} $",
            "daily_pnl_raw":   daily_pnl,
        }
    except Exception as e:
        return {"error": str(e)}


def status_broadcaster():
    """Sendet alle 10 Sekunden den Kontostatus an alle Browser."""
    while True:
        time.sleep(10)
        data = get_status_data()
        push_event("status", data)


threading.Thread(target=status_broadcaster, daemon=True).start()


# ─── Routen ──────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stream")
def stream():
    """SSE Endpoint - Browser verbindet sich hier für Live Updates."""
    def event_generator():
        q = queue.Queue(maxsize=20)
        with sse_lock:
            sse_clients.append(q)
        # Sofort aktuellen Status senden
        data = get_status_data()
        yield f"event: status\ndata: {json.dumps(data)}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with sse_lock:
                sse_clients.remove(q)

    return Response(event_generator(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/webhook", methods=["POST"])
def webhook():
    global skipped_today
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Kein JSON empfangen"}), 400

    if not verify_secret(data):
        return jsonify({"error": "Ungültiges Secret"}), 403

    symbol = data.get("symbol", "").upper()
    signal = data.get("signal", "").lower()
    rsi    = float(data.get("rsi", 50))
    ema    = float(data.get("ema", 0))
    atr    = data.get("atr")

    if not symbol or signal not in ("buy", "sell"):
        return jsonify({"error": "Fehlende Felder: symbol, signal (buy/sell)"}), 400

    # Symbol-Mapping (z.B. XAUUSD → GLD, SPX → SPY)
    with open("config.json") as f:
        cfg = json.load(f)
    tv_symbol = symbol
    symbol = cfg.get("symbol_map", {}).get(symbol, symbol)
    if tv_symbol != symbol:
        print(f"Symbol gemappt: {tv_symbol} → {symbol}")

    # 1. Tägliches Verlustlimit
    if not broker.check_daily_loss_limit():
        skipped_today += 1
        reason = "Tägliches Verlustlimit erreicht"
        push_event("skip", {"symbol": symbol, "signal": signal, "reason": reason})
        return jsonify({"status": "skip", "reason": reason}), 200

    # 2. Max. offene Positionen prüfen
    max_pos = cfg["risk_management"]["max_open_positions"]
    open_positions = broker.get_open_positions()
    open_symbols = [p.symbol for p in open_positions]

    if len(open_positions) >= max_pos:
        skipped_today += 1
        reason = f"Max. Positionen erreicht ({max_pos})"
        push_event("skip", {"symbol": symbol, "signal": signal, "reason": reason})
        return jsonify({"status": "skip", "reason": reason}), 200

    # Bereits eine Position in diesem Symbol?
    if symbol in open_symbols and signal == "buy":
        skipped_today += 1
        reason = f"Position in {symbol} bereits offen"
        push_event("skip", {"symbol": symbol, "signal": signal, "reason": reason})
        return jsonify({"status": "skip", "reason": reason}), 200

    # 2. Echten Preis von Alpaca holen
    try:
        entry_price = broker.get_current_price(symbol)
        if ema == 0:
            ema = entry_price
    except Exception as e:
        push_event("error_event", {"message": f"Preis Fehler {symbol}: {e}"})
        return jsonify({"error": f"Preis konnte nicht geholt werden: {e}"}), 500

    # 3. XGBoost Filter (nur für Symbole auf der ml_symbols Liste)
    ml_symbols = cfg.get("ml_symbols", ["AAPL"])
    atr_val = float(atr) if atr else entry_price * 0.01
    if symbol in ml_symbols and not ml_confirms_signal(rsi, entry_price, ema, atr_val, signal):
        skipped_today += 1
        reason = "XGBoost lehnt Signal ab"
        push_event("skip", {"symbol": symbol, "signal": signal, "reason": reason})
        return jsonify({"status": "skip", "reason": reason}), 200
    elif symbol not in ml_symbols:
        print(f"ML-Filter übersprungen für {symbol} (nur RSI-Signal)")

    # 4. Risikomanagement
    account_value = broker.get_account_value()
    trade_params = risk_manager.calculate_trade(
        entry_price=entry_price,
        signal=signal,
        account_value=account_value,
        atr=float(atr) if atr else None,
    )

    # Kaufkraft prüfen
    needed = int(trade_params["quantity"]) * entry_price
    available = broker.get_buying_power()
    if needed > available:
        skipped_today += 1
        reason = f"Nicht genug Kaufkraft ({available:,.0f}$ verfügbar, {needed:,.0f}$ benötigt)"
        push_event("skip", {"symbol": symbol, "signal": signal, "reason": reason})
        return jsonify({"status": "skip", "reason": reason}), 200

    # 5. Order senden
    order = broker.place_order(
        symbol=symbol,
        signal=signal,
        quantity=trade_params["quantity"],
        sl_price=trade_params["sl_price"],
        tp_price=trade_params["tp_price"],
    )

    log_entry = {**trade_params, **order, "account_value": account_value}

    # 6. Google Sheets
    try:
        sheets_logger.log_trade(log_entry)
    except Exception as e:
        print(f"Sheets Logging Fehler (Trade trotzdem ausgeführt): {e}")

    trades_today.append(log_entry)

    # 7. Live Dashboard updaten
    push_event("trade", {
        "time":   datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol,
        "signal": signal,
        "entry":  trade_params["entry_price"],
        "sl":     trade_params["sl_price"],
        "tp":     trade_params["tp_price"],
        "risk":   f"{trade_params['risk_percent']}% = {trade_params['risk_amount']}$",
        "rr":     trade_params["rr_ratio"],
    })
    push_event("status", get_status_data())

    return jsonify({
        "status": "order_sent",
        "symbol": symbol,
        "signal": signal,
        "entry":  trade_params["entry_price"],
        "sl":     trade_params["sl_price"],
        "tp":     trade_params["tp_price"],
        "risk":   f"{trade_params['risk_percent']}% = {trade_params['risk_amount']}$",
        "order_id": order["order_id"],
    }), 200


@app.route("/eod", methods=["POST"])
def end_of_day():
    data = request.get_json(silent=True) or {}
    if not verify_secret(data):
        return jsonify({"error": "Ungültiges Secret"}), 403

    account_value = broker.get_account_value()
    client = broker.get_client()
    account = client.get_account()
    daily_pnl = float(account.equity) - float(account.last_equity)

    try:
        sheets_logger.log_eod_summary(account_value, daily_pnl, len(trades_today))
    except Exception as e:
        return jsonify({"error": f"Sheets Fehler: {e}"}), 500

    trades_today.clear()
    return jsonify({
        "status": "Tagesabschluss geloggt",
        "account_value": f"{account_value:.2f}$",
        "daily_pnl": f"{daily_pnl:+.2f}$",
    })


@app.route("/status", methods=["GET"])
def status():
    data = get_status_data()
    data["model_loaded"] = model_loaded
    return jsonify(data)


if __name__ == "__main__":
    with open("config.json") as f:
        cfg = json.load(f)
    load_todays_trades()
    print("Bot gestartet - Warte auf TradingView Webhooks...")
    print(f"Dashboard: http://localhost:{cfg['server']['port']}")
    app.run(
        host=cfg["server"]["host"],
        port=cfg["server"]["port"],
        debug=False,
        threaded=True,
    )
