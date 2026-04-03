"""
TradingView RSI Bot - Webhook Server mit Live Dashboard
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
from scanner import Scanner

load_dotenv()

app = Flask(__name__)
trades_today  = []
skipped_today = 0
scanner_status = {"status": "Startet...", "last_scan": "–"}

# Separate ML-Modelle pro Symbol laden
models = {}
for sym in ("AAPL", "GLD", "SPY"):
    m = xgb.XGBClassifier()
    try:
        m.load_model(f"model_{sym}.json")
        models[sym] = m
        print(f"Modell geladen: model_{sym}.json")
    except Exception:
        print(f"WARNUNG: Kein Modell für {sym}")

# SSE
sse_clients = []
sse_lock    = threading.Lock()


def push_event(event_type: str, data: dict):
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        for q in sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


def verify_secret(data: dict) -> bool:
    return data.get("secret", "") == os.getenv("WEBHOOK_SECRET", "")


def ml_confirms_signal(symbol: str, rsi: float, entry_price: float,
                       ema: float, atr: float, signal: str) -> bool:
    model = models.get(symbol)
    if not model:
        return True
    ema200_dist = (entry_price - ema) / ema * 100 if ema > 0 else 0
    ema50_dist  = ema200_dist * 0.5
    vol_ratio   = 1.0  # Placeholder — echtes Volumen kommt vom Volumen-Check
    df = pd.DataFrame(
        [[rsi, ema200_dist, ema50_dist, atr, vol_ratio]],
        columns=["RSI", "EMA200_dist", "EMA50_dist", "ATR", "Vol_ratio"]
    )
    pred = model.predict(df)[0]
    confirmed = (signal == "buy" and pred == 1) or (signal == "sell" and pred == 0)
    print(f"XGBoost {symbol}: {'OK' if confirmed else 'ABLEHNT'} {signal.upper()} | RSI={rsi:.1f}")
    return confirmed


def calc_stats() -> dict:
    """Berechnet Win-Rate und Profit-Factor aus heutigen Trades."""
    wins = losses = gross_profit = gross_loss = 0
    for t in trades_today:
        pnl = t.get("pnl", 0)
        if pnl > 0:
            wins        += 1
            gross_profit += pnl
        elif pnl < 0:
            losses      += 1
            gross_loss  += abs(pnl)
    total = wins + losses
    win_rate      = round(wins / total * 100, 1) if total > 0 else 0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0
    return {"win_rate": win_rate, "profit_factor": profit_factor,
            "wins": wins, "losses": losses}


def get_status_data() -> dict:
    try:
        value     = broker.get_account_value()
        positions = broker.get_open_positions()
        account   = broker.get_client().get_account()
        daily_pnl = float(account.equity) - float(account.last_equity)

        positions_list = [{
            "symbol":  p.symbol,
            "side":    str(p.side.value),
            "qty":     float(p.qty),
            "entry":   round(float(p.avg_entry_price), 2),
            "current": round(float(p.current_price), 2),
            "pnl":     round(float(p.unrealized_pl or 0), 2),
            "pnl_pct": round(float(p.unrealized_plpc or 0) * 100, 2),
        } for p in positions]

        stats = calc_stats()
        return {
            "account_value":  f"{value:,.2f} $",
            "open_positions": len(positions),
            "positions_list": positions_list,
            "trades_today":   len(trades_today),
            "skipped_today":  skipped_today,
            "daily_pnl":      f"{daily_pnl:+.2f} $",
            "daily_pnl_raw":  daily_pnl,
            "win_rate":       stats["win_rate"],
            "profit_factor":  stats["profit_factor"],
            "wins":           stats["wins"],
            "losses":         stats["losses"],
            "scanner":        scanner_status,
        }
    except Exception as e:
        return {"error": str(e)}


def load_todays_trades():
    global trades_today
    try:
        client = broker.get_client()
        today  = datetime.now().strftime("%Y-%m-%d")
        count  = 0
        for o in client.get_orders():
            if str(o.created_at)[:10] == today and \
               str(o.status) not in ("canceled", "expired", "rejected"):
                trades_today.append({
                    "symbol":      str(o.symbol),
                    "signal":      str(o.side.value),
                    "order_id":    str(o.id),
                    "quantity":    float(o.qty or 0),
                    "entry_price": float(o.filled_avg_price or 0),
                    "pnl":         0,
                    "status":      str(o.status),
                })
                count += 1
        if count:
            print(f"Startup: {count} heutige Order(s) geladen.")
    except Exception as e:
        print(f"Startup Fehler: {e}")


def status_broadcaster():
    while True:
        time.sleep(10)
        push_event("status", get_status_data())


threading.Thread(target=status_broadcaster, daemon=True).start()


def execute_trade(symbol: str, signal: str, atr: float, **_):
    """Wird vom Scanner aufgerufen — führt dieselbe Pipeline wie der Webhook aus."""
    global skipped_today

    with open("config.json") as f:
        cfg = json.load(f)

    def skip(reason):
        global skipped_today
        skipped_today += 1
        push_event("skip", {"symbol": symbol, "signal": signal, "reason": reason})
        print(f"[Scanner] Skip {symbol}: {reason}")

    open_positions = broker.get_open_positions()
    open_symbols   = [p.symbol for p in open_positions]

    if len(open_positions) >= cfg["risk_management"]["max_open_positions"]:
        return skip(f"Max. Positionen erreicht")
    if symbol in open_symbols and signal == "buy":
        return skip(f"Position in {symbol} bereits offen")

    try:
        entry_price = broker.get_current_price(symbol)
    except Exception as e:
        return skip(f"Preis Fehler: {e}")

    ok, reason = broker.check_volume(symbol)
    if not ok:
        return skip(reason)

    account_value = broker.get_account_value()
    trade_params  = risk_manager.calculate_trade(entry_price, signal, account_value, atr=atr)

    needed = int(trade_params["quantity"]) * entry_price
    if needed > broker.get_buying_power():
        return skip(f"Nicht genug Kaufkraft")

    order = broker.place_order(symbol, signal, trade_params["quantity"],
                                trade_params["sl_price"], trade_params["tp_price"])

    log_entry = {**trade_params, **order, "account_value": account_value, "pnl": 0}
    try:
        sheets_logger.log_trade(log_entry)
    except Exception as e:
        print(f"Sheets Fehler: {e}")

    trades_today.append(log_entry)
    push_event("trade", {
        "time":   datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol, "signal": signal,
        "entry":  trade_params["entry_price"],
        "sl":     trade_params["sl_price"],
        "tp":     trade_params["tp_price"],
        "risk":   f"{trade_params['risk_percent']}% = {trade_params['risk_amount']}$",
        "rr":     trade_params["rr_ratio"],
    })
    push_event("status", get_status_data())
    print(f"[Scanner] Trade ausgeführt: {signal.upper()} {symbol} @ {entry_price}")


def start_scanner():
    def push_with_scanner(event_type, data):
        global scanner_status
        if event_type == "scanner":
            scanner_status = data
        push_event(event_type, data)

    s = Scanner(execute_fn=execute_trade, push_fn=push_with_scanner)
    s.run()


threading.Thread(target=start_scanner, daemon=True).start()


# ─── Routen ──────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stream")
def stream():
    def event_generator():
        q = queue.Queue(maxsize=20)
        with sse_lock:
            sse_clients.append(q)
        yield f"event: status\ndata: {json.dumps(get_status_data())}\n\n"
        try:
            while True:
                try:
                    yield q.get(timeout=25)
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
        return jsonify({"error": "Fehlende Felder: symbol, signal"}), 400

    with open("config.json") as f:
        cfg = json.load(f)

    # Symbol-Mapping
    tv_symbol = symbol
    symbol    = cfg.get("symbol_map", {}).get(symbol, symbol)
    if tv_symbol != symbol:
        print(f"Symbol gemappt: {tv_symbol} -> {symbol}")

    def skip(reason):
        global skipped_today
        skipped_today += 1
        push_event("skip", {"symbol": symbol, "signal": signal, "reason": reason})
        return jsonify({"status": "skip", "reason": reason}), 200

    # 1. Marktzeiten
    ok, reason = broker.check_market_hours()
    if not ok:
        return skip(reason)

    # 2. Tägliches Verlustlimit
    if not broker.check_daily_loss_limit():
        return skip("Tägliches Verlustlimit erreicht")

    # 3. Max. offene Positionen
    open_positions = broker.get_open_positions()
    open_symbols   = [p.symbol for p in open_positions]
    if len(open_positions) >= cfg["risk_management"]["max_open_positions"]:
        return skip(f"Max. Positionen erreicht ({cfg['risk_management']['max_open_positions']})")
    if symbol in open_symbols and signal == "buy":
        return skip(f"Position in {symbol} bereits offen")

    # 4. Echten Preis holen
    try:
        entry_price = broker.get_current_price(symbol)
        if ema == 0:
            ema = entry_price
    except Exception as e:
        push_event("error_event", {"message": f"Preis Fehler {symbol}: {e}"})
        return jsonify({"error": str(e)}), 500

    # 5. Volumen-Filter
    ok, reason = broker.check_volume(symbol)
    if not ok:
        return skip(reason)

    # 6. Multi-Timeframe Check
    ok, reason = broker.check_multiframe_rsi(symbol, signal)
    if not ok:
        return skip(reason)

    # 7. ML-Filter
    atr_val = float(atr) if atr else entry_price * 0.01
    if symbol in cfg.get("ml_symbols", []):
        if not ml_confirms_signal(symbol, rsi, entry_price, ema, atr_val, signal):
            return skip("XGBoost lehnt Signal ab")

    # 8. Risikomanagement
    account_value = broker.get_account_value()
    trade_params  = risk_manager.calculate_trade(entry_price, signal, account_value,
                                                  atr=atr_val)

    # 9. Kaufkraft prüfen
    needed    = int(trade_params["quantity"]) * entry_price
    available = broker.get_buying_power()
    if needed > available:
        return skip(f"Nicht genug Kaufkraft ({available:,.0f}$ vs {needed:,.0f}$)")

    # 10. Order senden
    order = broker.place_order(symbol, signal, trade_params["quantity"],
                                trade_params["sl_price"], trade_params["tp_price"])

    log_entry = {**trade_params, **order, "account_value": account_value, "pnl": 0}

    try:
        sheets_logger.log_trade(log_entry)
    except Exception as e:
        print(f"Sheets Fehler: {e}")

    trades_today.append(log_entry)

    push_event("trade", {
        "time":   datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol, "signal": signal,
        "entry":  trade_params["entry_price"],
        "sl":     trade_params["sl_price"],
        "tp":     trade_params["tp_price"],
        "risk":   f"{trade_params['risk_percent']}% = {trade_params['risk_amount']}$",
        "rr":     trade_params["rr_ratio"],
    })
    push_event("status", get_status_data())

    return jsonify({
        "status":   "order_sent",
        "symbol":   symbol,
        "signal":   signal,
        "entry":    trade_params["entry_price"],
        "sl":       trade_params["sl_price"],
        "tp":       trade_params["tp_price"],
        "trail":    f"{cfg['risk_management'].get('trailing_stop_percent', 1.5)}%",
        "order_id": order["order_id"],
    }), 200


@app.route("/eod", methods=["POST"])
def end_of_day():
    data = request.get_json(silent=True) or {}
    if not verify_secret(data):
        return jsonify({"error": "Ungültiges Secret"}), 403
    account_value = broker.get_account_value()
    account       = broker.get_client().get_account()
    daily_pnl     = float(account.equity) - float(account.last_equity)
    try:
        sheets_logger.log_eod_summary(account_value, daily_pnl, len(trades_today))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    trades_today.clear()
    return jsonify({"status": "ok", "pnl": f"{daily_pnl:+.2f}$"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(get_status_data())


@app.route("/history", methods=["GET"])
def history():
    """Gibt geschlossene Trades der letzten 30 Tage zurück."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        import pytz
        from datetime import timedelta

        client = broker.get_client()
        ny     = pytz.timezone("America/New_York")
        start  = datetime.now(ny) - timedelta(days=30)

        req    = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=start,
            limit=100,
        )
        orders = client.get_orders(req)

        result = []
        for o in orders:
            if str(o.status) != "OrderStatus.filled":
                continue
            filled = float(o.filled_avg_price or 0)
            qty    = float(o.qty or 0)
            side   = str(o.side.value)

            # PnL schätzen: für verkaufte Positionen
            # (exakter PnL kommt aus Alpaca Activities)
            result.append({
                "date":     str(o.filled_at)[:10] if o.filled_at else str(o.created_at)[:10],
                "time":     str(o.filled_at)[11:16] if o.filled_at else "–",
                "symbol":   o.symbol,
                "side":     side,
                "qty":      qty,
                "price":    round(filled, 2),
                "value":    round(filled * qty, 2),
                "order_id": str(o.id),
            })

        return jsonify({"trades": result[::-1]})  # Neueste zuerst
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    with open("config.json") as f:
        cfg = json.load(f)
    load_todays_trades()
    print("Bot gestartet!")
    print(f"Dashboard: http://localhost:{cfg['server']['port']}")
    app.run(host=cfg["server"]["host"], port=cfg["server"]["port"],
            debug=False, threaded=True)
