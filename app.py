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
import base64
import hmac as hmac_lib
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, render_template, Response, session, redirect
from dotenv import load_dotenv
import xgboost as xgb
import pandas as pd

import broker
import risk_manager
import sheets_logger
from scanner import Scanner

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
app.permanent_session_lifetime = timedelta(days=30)

# Einfaches In-Memory Rate-Limiting für Login-Versuche
_login_attempts: dict = {}   # {ip: [timestamp, ...]}
_LOGIN_MAX     = 5           # Max. Versuche
_LOGIN_WINDOW  = 300         # Innerhalb von 5 Minuten


def _rate_limit_ok(ip: str) -> bool:
    now = time.time()
    attempts = [t for t in _login_attempts.get(ip, []) if now - t < _LOGIN_WINDOW]
    _login_attempts[ip] = attempts
    if len(attempts) >= _LOGIN_MAX:
        return False
    _login_attempts[ip].append(now)
    return True


def _check_credentials(password: str) -> bool:
    expected = os.getenv("DASHBOARD_PASSWORD", "")
    if not expected:
        return True   # Kein Passwort gesetzt → Entwicklungsmodus
    return hmac_lib.compare_digest(password, expected)


def require_auth(f):
    """Session-Cookie Auth — fällt zurück auf HTTP Basic Auth für API-Zugriffe."""
    @wraps(f)
    def decorated(*args, **kwargs):
        expected_pass = os.getenv("DASHBOARD_PASSWORD", "")
        if not expected_pass:
            return f(*args, **kwargs)

        # 1. Session-Cookie (Browser / iPhone nach Login)
        if session.get("auth"):
            return f(*args, **kwargs)

        # 2. HTTP Basic Auth (curl / API-Zugriffe)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                decoded  = base64.b64decode(auth[6:]).decode("utf-8")
                _, pw    = decoded.split(":", 1)
                if _check_credentials(pw):
                    return f(*args, **kwargs)
            except Exception:
                pass

        # Für API-Anfragen (JSON erwartet) → 401
        if request.accept_mimetypes.best == "application/json":
            return Response("Nicht autorisiert", 401)

        # Für Browser → Login-Seite
        return redirect(f"/login?next={request.path}")
    return decorated
trades_today  = []
skipped_today = 0
scanner_status = {"status": "Startet...", "last_scan": "–"}

# ── Caches ────────────────────────────────────────────────────────────────────
_status_cache  = {"data": None, "ts": 0.0}   # pre-computed, aktualisiert alle 10s
_stats_cache   = {"data": None, "ts": 0.0}   # calc_stats:      5 Min TTL
_weekly_cache  = {"data": None, "ts": 0.0}   # get_weekly_stats: 10 Min TTL
_history_cache = {"data": None, "ts": 0.0}   # /history:         2 Min TTL
_cache_lock    = threading.Lock()

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


def _calc_portfolio_heat(open_positions: list, sl_tp_map: dict, account_value: float) -> float:
    """Gesamtrisiko aller offenen Positionen in % des Kontos."""
    total_risk = 0.0
    for p in open_positions:
        sym   = p.symbol
        entry = float(p.avg_entry_price)
        qty   = float(p.qty)
        sl    = sl_tp_map.get(sym, {}).get("sl")
        if sl:
            total_risk += abs(entry - sl) * qty
    return total_risk / account_value * 100 if account_value > 0 else 0.0


def _get_consecutive_losses() -> int:
    """Zählt aufeinanderfolgende Verluste aus den letzten abgeschlossenen Trades."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        import pytz

        client  = broker.get_client()
        ny      = pytz.timezone("America/New_York")
        since   = datetime.now(ny) - timedelta(days=14)
        req     = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=since, limit=30)
        orders  = client.get_orders(req)
        filled  = [o for o in orders
                   if "filled" in str(o.status).lower() and o.filled_avg_price]
        if not filled:
            return 0

        buy_queues: dict = {}
        pnls: list = []
        for o in reversed(filled):
            sym   = o.symbol
            price = float(o.filled_avg_price)
            qty   = float(o.qty or 1)
            if o.side.value == "buy":
                buy_queues.setdefault(sym, []).append({"price": price, "qty": qty})
            else:
                q      = buy_queues.get(sym, [])
                pnl    = 0.0
                remain = qty
                while remain > 0 and q:
                    b     = q[0]
                    match = min(b["qty"], remain)
                    pnl   += (price - b["price"]) * match
                    b["qty"] -= match
                    remain   -= match
                    if b["qty"] <= 0:
                        q.pop(0)
                pnls.append(pnl)

        consecutive = 0
        for p in reversed(pnls):
            if p < 0:
                consecutive += 1
            else:
                break
        return consecutive
    except Exception:
        return 0


def calc_stats() -> dict:
    """Win-Rate + Profit-Factor aus Alpaca-Aktivitäten (letzte 30 Tage). Cache: 5 Min."""
    global _stats_cache
    if _stats_cache["data"] and time.time() - _stats_cache["ts"] < 300:
        return _stats_cache["data"]
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        import pytz
        from datetime import timedelta

        client = broker.get_client()
        ny     = pytz.timezone("America/New_York")
        start  = datetime.now(ny) - timedelta(days=30)
        req    = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=start, limit=200)
        orders = client.get_orders(req)

        # FIFO PnL pro Symbol berechnen
        buy_queues = {}
        trade_pnls = []
        for o in reversed(orders):
            if "filled" not in str(o.status).lower():
                continue
            sym   = o.symbol
            price = float(o.filled_avg_price or 0)
            qty   = float(o.qty or 0)
            if o.side.value == "buy":
                buy_queues.setdefault(sym, []).append({"price": price, "qty": qty})
            else:
                remain = qty
                pnl    = 0.0
                queue  = buy_queues.get(sym, [])
                while remain > 0 and queue:
                    b = queue[0]
                    match = min(b["qty"], remain)
                    pnl    += (price - b["price"]) * match
                    b["qty"] -= match
                    remain   -= match
                    if b["qty"] <= 0:
                        queue.pop(0)
                trade_pnls.append(round(pnl, 2))

        wins         = sum(1 for p in trade_pnls if p > 0)
        losses       = sum(1 for p in trade_pnls if p < 0)
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss   = abs(sum(p for p in trade_pnls if p < 0))
        total        = wins + losses
        result = {
            "win_rate":      round(wins / total * 100, 1) if total > 0 else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            "wins":          wins,
            "losses":        losses,
        }
        _stats_cache["data"] = result
        _stats_cache["ts"]   = time.time()
        return result
    except Exception:
        return {"win_rate": 0, "profit_factor": 0, "wins": 0, "losses": 0}


def get_weekly_stats() -> dict:
    """Konto-Verlauf der letzten 7 Tage direkt via Alpaca REST API. Cache: 10 Min."""
    global _weekly_cache
    if _weekly_cache["data"] and time.time() - _weekly_cache["ts"] < 600:
        return _weekly_cache["data"]
    try:
        import requests as req_lib
        import pytz
        headers = {
            "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY"),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
        }
        resp = req_lib.get(
            "https://paper-api.alpaca.markets/v2/account/portfolio/history",
            params={"period": "1W", "timeframe": "1D"},
            headers=headers, timeout=10,
        )
        resp.raise_for_status()
        data       = resp.json()
        timestamps = data.get("timestamp", [])
        equity     = data.get("equity", [])
        pnl_list   = data.get("profit_loss", [])

        days = []
        for i, ts in enumerate(timestamps):
            dt = datetime.fromtimestamp(ts, tz=pytz.UTC)
            days.append({
                "date":   dt.strftime("%a %d.%m."),
                "equity": round(float(equity[i]), 2) if i < len(equity) else 0,
                "pnl":    round(float(pnl_list[i]), 2) if i < len(pnl_list) else 0,
            })

        total_week_pnl = round(sum(d["pnl"] for d in days), 2)
        result = {"days": days, "total_week_pnl": total_week_pnl}
        _weekly_cache["data"] = result
        _weekly_cache["ts"]   = time.time()
        return result
    except Exception:
        return {"days": [], "total_week_pnl": 0}


def get_status_data() -> dict:
    try:
        value     = broker.get_account_value()
        positions = broker.get_open_positions()
        account   = broker.get_client().get_account()
        daily_pnl = float(account.equity) - float(account.last_equity)

        sl_tp = broker.get_sl_tp_map()
        positions_list = [{
            "symbol":  p.symbol,
            "side":    str(p.side.value),
            "qty":     float(p.qty),
            "entry":   round(float(p.avg_entry_price), 2),
            "current": round(float(p.current_price), 2),
            "pnl":     round(float(p.unrealized_pl or 0), 2),
            "pnl_pct": round(float(p.unrealized_plpc or 0) * 100, 2),
            "sl":      sl_tp.get(p.symbol, {}).get("sl"),
            "tp":      sl_tp.get(p.symbol, {}).get("tp"),
        } for p in positions]

        stats = calc_stats()
        buys  = sum(1 for t in trades_today if t.get("signal") == "buy")
        sells = sum(1 for t in trades_today if t.get("signal") == "sell")

        # Live-Konto-Info (nur wenn aktiviert)
        live_info = None
        try:
            with open("config.json") as f:
                _cfg = json.load(f)
            if _cfg.get("live_trading", {}).get("enabled", False):
                la = broker.get_live_account()
                lp = broker.get_live_open_positions()
                live_info = {
                    "value":     f"{la['portfolio_value']:,.2f} $",
                    "daily_pnl": f"{la['daily_pnl']:+.2f} $",
                    "positions": len(lp),
                }
        except Exception:
            pass

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
            "signals_buy":    buys,
            "signals_sell":   sells,
            "signals_skip":   skipped_today,
            "scanner":        scanner_status,
            "weekly":         get_weekly_stats(),
            "live":           live_info,
        }
    except Exception as e:
        return {"error": str(e)}


def load_todays_trades():
    global trades_today
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        import pytz

        client     = broker.get_client()
        ny         = pytz.timezone("America/New_York")
        today_str  = datetime.now(ny).strftime("%Y-%m-%d")
        today_start = datetime.now(ny).replace(hour=0, minute=0, second=0, microsecond=0)

        # ALL holt sowohl offene als auch geschlossene Orders
        req    = GetOrdersRequest(status=QueryOrderStatus.ALL, after=today_start, limit=200)
        orders = client.get_orders(req)
        count  = 0
        for o in orders:
            status = str(o.status).lower()
            if status in ("canceled", "expired", "rejected"):
                continue
            # Nur Parent-Orders (keine Bracket-Legs)
            if o.legs:
                pass  # Parent-Order mit Legs → gehört rein
            elif getattr(o, "order_class", None) and str(o.order_class) != "OrderClass.simple":
                continue  # Bracket-Leg → überspringen
            created = str(o.created_at)[:10] if o.created_at else ""
            if created != today_str:
                continue
            trades_today.append({
                "symbol":      str(o.symbol),
                "signal":      str(o.side.value),
                "order_id":    str(o.id),
                "quantity":    float(o.qty or 0),
                "entry_price": float(o.filled_avg_price or 0),
                "pnl":         0,
                "status":      status,
                "time":        str(o.created_at)[11:16] if o.created_at else "–",
            })
            count += 1
        if count:
            print(f"Startup: {count} heutige Trade(s) geladen.")
        else:
            print("Startup: Keine heutigen Trades gefunden.")
    except Exception as e:
        print(f"Startup Fehler: {e}")


def status_broadcaster():
    global _status_cache
    while True:
        try:
            data = get_status_data()
            with _cache_lock:
                _status_cache["data"] = data
                _status_cache["ts"]   = time.time()
            push_event("status", data)
        except Exception as e:
            print(f"[Broadcaster] Fehler: {e}")
        time.sleep(10)


def get_cached_status() -> dict:
    """Gibt gecachten Status zurück — berechnet neu falls Cache leer."""
    with _cache_lock:
        if _status_cache["data"]:
            return _status_cache["data"]
    return get_status_data()


threading.Thread(target=status_broadcaster, daemon=True).start()


def execute_trade(symbol: str, signal: str, atr: float, score: float = None, price: float = None, **_):
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
        return skip("Max. Positionen erreicht")
    if symbol in open_symbols and signal == "buy":
        return skip(f"Position in {symbol} bereits offen")

    try:
        entry_price = price if price else broker.get_current_price(symbol)
    except Exception as e:
        return skip(f"Preis Fehler: {e}")

    ok, reason = broker.check_volume(symbol)
    if not ok:
        return skip(reason)

    account_value = broker.get_account_value()

    # Fix 1: Portfolio-Wärme — Gesamtrisiko aller Positionen prüfen
    max_heat = cfg["risk_management"].get("max_portfolio_heat_percent", 3.0)
    if max_heat > 0:
        sl_tp_map    = broker.get_sl_tp_map()
        current_heat = _calc_portfolio_heat(open_positions, sl_tp_map, account_value)
        if current_heat >= max_heat:
            return skip(f"Portfolio-Wärme {current_heat:.1f}% ≥ {max_heat}% — kein neuer Trade")

    # Fix 3: Verlust-Serie — Positionsgröße nach mehreren Verlusten reduzieren
    risk_pct_override = None
    consec_losses = _get_consecutive_losses()
    loss_limit    = cfg["risk_management"].get("consecutive_loss_limit", 3)
    size_factor   = cfg["risk_management"].get("consecutive_loss_size_factor", 0.5)
    base_risk     = cfg["risk_management"]["risk_per_trade_percent"] / 100
    if consec_losses >= loss_limit + 2:
        return skip(f"{consec_losses} Verluste in Folge — Trading pausiert")
    elif consec_losses >= loss_limit:
        risk_pct_override = base_risk * size_factor
        print(f"[Risk] {consec_losses} Verluste in Folge — Größe auf {size_factor*100:.0f}% reduziert")

    trade_params = risk_manager.calculate_trade(
        entry_price, signal, account_value, atr=atr, score=score,
        risk_pct_override=risk_pct_override,
    )

    needed = int(trade_params["quantity"]) * entry_price
    if needed > broker.get_buying_power():
        return skip("Nicht genug Kaufkraft")

    order = broker.place_order(symbol, signal, trade_params["quantity"],
                                trade_params["sl_price"], trade_params["tp_price"])

    log_entry = {**trade_params, **order, "account_value": account_value, "pnl": 0}
    try:
        sheets_logger.log_trade(log_entry)
    except Exception as e:
        print(f"Sheets Fehler: {e}")

    trades_today.append(log_entry)
    score_str = f" | Score={score}" if score else ""
    push_event("trade", {
        "time":   datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol, "signal": signal,
        "entry":  trade_params["entry_price"],
        "sl":     trade_params["sl_price"],
        "tp":     trade_params["tp_price"],
        "risk":   f"{trade_params['risk_percent']}% = {trade_params['risk_amount']}$",
        "rr":     trade_params["rr_ratio"],
        "score":  score,
    })
    push_event("status", get_status_data())
    print(f"[Scanner] Trade ausgeführt: {signal.upper()} {symbol} @ {entry_price}{score_str}")

    # Live-Trade spiegeln (falls aktiviert)
    execute_live_trade(symbol, signal, atr)


def execute_live_trade(symbol: str, signal: str, atr: float):
    """
    Spiegelt einen genehmigten Trade auf dem Live-Konto mit kleiner Position.
    Wird nach dem Paper-Trade aufgerufen — alle Filter wurden bereits geprüft.
    """
    with open("config.json") as f:
        cfg = json.load(f)

    live_cfg = cfg.get("live_trading", {})
    if not live_cfg.get("enabled", False):
        return

    try:
        # Live-Konto prüfen
        live_positions = broker.get_live_open_positions()
        if len(live_positions) >= live_cfg.get("max_open_positions", 2):
            print(f"[Live] Skip {symbol}: Max. {live_cfg['max_open_positions']} Positionen erreicht")
            return

        live_open_syms = [p.symbol for p in live_positions]
        if symbol in live_open_syms and signal == "buy":
            print(f"[Live] Skip {symbol}: Position bereits offen")
            return

        if not broker.check_live_daily_loss_limit():
            return

        # Positionsgröße mit kleinerem Risiko berechnen
        live_acc_info   = broker.get_live_account()
        live_risk_pct   = live_cfg.get("risk_per_trade_percent", 0.25) / 100
        entry_price     = broker.get_current_price(symbol)

        live_params = risk_manager.calculate_trade(
            entry_price, signal, live_acc_info["portfolio_value"],
            atr=atr, risk_pct_override=live_risk_pct,
        )

        # Kaufkraft prüfen
        needed = int(live_params["quantity"]) * entry_price
        if needed > live_acc_info["buying_power"]:
            print(f"[Live] Skip {symbol}: Nicht genug Kaufkraft "
                  f"({live_acc_info['buying_power']:.0f}$ < {needed:.0f}$)")
            return

        # Order aufgeben
        broker.place_live_order(
            symbol, signal,
            live_params["quantity"],
            live_params["sl_price"],
            live_params["tp_price"],
        )

        push_event("live_trade", {
            "time":   datetime.now().strftime("%H:%M:%S"),
            "symbol": symbol,
            "signal": signal,
            "entry":  live_params["entry_price"],
            "sl":     live_params["sl_price"],
            "tp":     live_params["tp_price"],
            "qty":    int(live_params["quantity"]),
            "risk":   f"{live_risk_pct * 100:.2f}% = {live_params['risk_amount']:.2f}$",
        })

    except Exception as e:
        print(f"[Live] Fehler bei {symbol}: {e}")
        push_event("skip", {"symbol": symbol, "signal": signal,
                             "reason": f"Live-Order fehlgeschlagen: {e}"})


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

_LOGIN_HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="Trading Bot">
  <title>Trading Bot – Login</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f1117; display: flex; align-items: center;
           justify-content: center; min-height: 100vh; font-family: -apple-system, sans-serif; }
    .card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 16px;
            padding: 40px 32px; width: 100%; max-width: 360px; }
    h1 { color: #e2e8f0; font-size: 1.4rem; margin-bottom: 8px; text-align: center; }
    p  { color: #64748b; font-size: 0.85rem; text-align: center; margin-bottom: 28px; }
    label { color: #94a3b8; font-size: 0.8rem; display: block; margin-bottom: 6px; }
    input[type=password] {
      width: 100%; padding: 12px 14px; background: #0f1117; border: 1px solid #2a2d3a;
      border-radius: 8px; color: #e2e8f0; font-size: 1rem; outline: none;
      transition: border-color .2s;
    }
    input[type=password]:focus { border-color: #3b82f6; }
    button {
      width: 100%; margin-top: 18px; padding: 13px;
      background: #3b82f6; border: none; border-radius: 8px;
      color: #fff; font-size: 1rem; font-weight: 600; cursor: pointer;
      transition: background .2s;
    }
    button:hover { background: #2563eb; }
    .error { background: #7f1d1d; color: #fca5a5; border-radius: 8px;
             padding: 10px 14px; font-size: 0.85rem; margin-bottom: 16px; text-align: center; }
    .icon { font-size: 2.5rem; text-align: center; margin-bottom: 14px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">📈</div>
    <h1>Trading Bot</h1>
    <p>Bitte einloggen um fortzufahren</p>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST">
      <label for="pw">Passwort</label>
      <input type="password" id="pw" name="password" autofocus autocomplete="current-password"
             placeholder="••••••••••••">
      <button type="submit">Einloggen</button>
    </form>
  </div>
</body>
</html>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        if not _rate_limit_ok(ip):
            return _login_page("Zu viele Versuche — bitte 5 Minuten warten")
        pw = request.form.get("password", "")
        if _check_credentials(pw):
            session.permanent = True
            session["auth"]   = True
            return redirect(request.args.get("next", "/"))
        return _login_page("Falsches Passwort — bitte erneut versuchen")
    if session.get("auth"):
        return redirect("/")
    return _login_page(None)


def _login_page(error):
    from flask import render_template_string
    return render_template_string(_LOGIN_HTML, error=error), (401 if error else 200)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@require_auth
def dashboard():
    return render_template("dashboard.html")


@app.route("/stream")
@require_auth
def stream():
    def event_generator():
        import screener as sc
        q = queue.Queue(maxsize=20)
        with sse_lock:
            sse_clients.append(q)

        # Beim Verbinden: gecachten Status sofort senden (kein API-Call nötig)
        yield f"event: status\ndata: {json.dumps(get_cached_status())}\n\n"

        # Screener-Ergebnisse wiederherstellen (falls bereits gelaufen)
        if sc.screener_results:
            yield f"event: screener\ndata: {json.dumps({'status': 'Letzter Run: ' + (sc.last_screen_time or '–'), 'results': sc.screener_results, 'last_scan': sc.last_screen_time, 'progress': 100})}\n\n"

        # Heutige Trades wiederherstellen
        if trades_today:
            yield f"event: trades_init\ndata: {json.dumps(trades_today)}\n\n"

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
@require_auth
def status():
    return jsonify(get_status_data())


@app.route("/live/test", methods=["GET"])
@require_auth
def live_test():
    """Testet die Live-API-Verbindung und gibt Konto-Info zurück."""
    try:
        acc = broker.get_live_account()
        return jsonify({
            "status":          "ok",
            "connected":       True,
            "portfolio_value": acc["portfolio_value"],
            "buying_power":    acc["buying_power"],
            "daily_pnl":       acc["daily_pnl"],
        })
    except ValueError as e:
        return jsonify({"status": "error", "connected": False,
                        "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "connected": False,
                        "message": f"Verbindungsfehler: {e}"}), 500


@app.route("/live/toggle", methods=["POST"])
@require_auth
def live_toggle():
    """Aktiviert / deaktiviert Live-Trading in config.json."""
    try:
        with open("config.json") as f:
            cfg = json.load(f)

        current = cfg.get("live_trading", {}).get("enabled", False)
        cfg.setdefault("live_trading", {})["enabled"] = not current

        # Vor dem Aktivieren: Verbindung prüfen
        if cfg["live_trading"]["enabled"]:
            try:
                broker.get_live_account()
            except Exception as e:
                return jsonify({"status": "error",
                                "message": f"Live-Keys ungültig: {e}"}), 400

        with open("config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        new_state = cfg["live_trading"]["enabled"]
        print(f"[Live] Trading {'AKTIVIERT' if new_state else 'DEAKTIVIERT'}")
        push_event("status", get_status_data())
        return jsonify({"status": "ok", "live_enabled": new_state})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/live/config", methods=["POST"])
@require_auth
def live_config():
    """Aktualisiert Live-Trading Parameter (Risiko, Max-Positionen etc.)."""
    data = request.get_json(silent=True) or {}
    try:
        with open("config.json") as f:
            cfg = json.load(f)

        lc = cfg.setdefault("live_trading", {})
        if "risk_per_trade_percent" in data:
            lc["risk_per_trade_percent"] = float(data["risk_per_trade_percent"])
        if "max_open_positions" in data:
            lc["max_open_positions"] = int(data["max_open_positions"])
        if "max_daily_loss_percent" in data:
            lc["max_daily_loss_percent"] = float(data["max_daily_loss_percent"])

        with open("config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        return jsonify({"status": "ok", "live_trading": lc})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/history", methods=["GET"])
@require_auth
def history():
    """Gibt geschlossene Trades der letzten 30 Tage zurück. Cache: 2 Min."""
    global _history_cache
    force = request.args.get("refresh") == "1"
    if not force and _history_cache["data"] and time.time() - _history_cache["ts"] < 120:
        return jsonify(_history_cache["data"])
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
            if "filled" not in str(o.status).lower():
                continue
            filled = float(o.filled_avg_price or 0)
            qty    = float(o.qty or 0)
            side   = str(o.side.value)

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

        payload = {"trades": result[::-1]}
        _history_cache["data"] = payload
        _history_cache["ts"]   = time.time()
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/equity")
@require_auth
def equity_history():
    """30-Tage Kontostand-Verlauf von Alpaca (Fix 9 — Equity-Kurve)."""
    try:
        import requests as req_lib
        import pytz
        headers = {
            "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY"),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
        }
        resp = req_lib.get(
            "https://paper-api.alpaca.markets/v2/account/portfolio/history",
            params={"period": "1M", "timeframe": "1D"},
            headers=headers, timeout=10,
        )
        resp.raise_for_status()
        data       = resp.json()
        timestamps = data.get("timestamp", [])
        equity_vals = data.get("equity", [])

        points = []
        for i, ts in enumerate(timestamps):
            val = float(equity_vals[i]) if i < len(equity_vals) else 0
            if val > 0:
                dt = datetime.fromtimestamp(ts, tz=pytz.UTC)
                points.append({"date": dt.strftime("%d.%m."), "value": round(val, 2)})
        return jsonify({"points": points})
    except Exception as e:
        return jsonify({"points": [], "error": str(e)})


@app.route("/backtest", methods=["POST"])
@require_auth
def run_backtest_endpoint():
    """Führt Backtest für ein oder mehrere Symbole durch."""
    data    = request.get_json(silent=True) or {}
    symbols = data.get("symbols", ["AAPL", "GLD", "SPY"])
    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    days        = int(data.get("days", 365))
    rr_ratio    = float(data.get("rr_ratio", 2.0))
    rsi_os      = int(data.get("rsi_oversold", 30))
    rsi_ob      = int(data.get("rsi_overbought", 70))
    vol         = float(data.get("vol_threshold", 0.8))
    max_hold    = int(data.get("max_hold_days", 20))

    try:
        import backtest as bt
        results = bt.run_multi_backtest(
            symbols, days=days, rr_ratio=rr_ratio,
            rsi_oversold=rsi_os, rsi_overbought=rsi_ob,
            vol_threshold=vol, max_hold_days=max_hold,
        )
        return jsonify(results)
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
