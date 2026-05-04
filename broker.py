"""
Alpaca Broker Wrapper - Paper Trading & Live Trading
"""

import os
import json
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest,
    GetOrdersRequest, TrailingStopOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockLatestQuoteRequest, StockLatestTradeRequest, StockBarsRequest
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()


def get_client() -> TradingClient:
    with open("config.json") as f:
        config = json.load(f)
    return TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=config["alpaca"]["paper_trading"],
    )


def get_live_client() -> TradingClient:
    """Live-Trading Client (echtes Geld!) — braucht ALPACA_LIVE_API_KEY in .env"""
    key    = os.getenv("ALPACA_LIVE_API_KEY")
    secret = os.getenv("ALPACA_LIVE_SECRET_KEY")
    if not key or not secret:
        raise ValueError("ALPACA_LIVE_API_KEY / ALPACA_LIVE_SECRET_KEY fehlen in .env")
    return TradingClient(api_key=key, secret_key=secret, paper=False)


def get_live_account() -> dict:
    """Gibt Live-Konto-Infos zurück (portfolio_value, equity, daily_pnl)."""
    acc = get_live_client().get_account()
    equity      = float(acc.equity)
    last_equity = float(acc.last_equity)
    return {
        "portfolio_value": float(acc.portfolio_value),
        "equity":          equity,
        "daily_pnl":       round(equity - last_equity, 2),
        "buying_power":    float(acc.buying_power),
    }


def get_live_open_positions() -> list:
    return get_live_client().get_all_positions()


def check_live_daily_loss_limit() -> bool:
    with open("config.json") as f:
        cfg = json.load(f)
    max_loss_pct = cfg["live_trading"]["max_daily_loss_percent"] / 100
    acc          = get_live_client().get_account()
    equity       = float(acc.equity)
    last_equity  = float(acc.last_equity)
    daily_pnl    = (equity - last_equity) / last_equity if last_equity > 0 else 0
    if daily_pnl <= -max_loss_pct:
        print(f"[Live] Tägliches Verlustlimit erreicht! PnL: {daily_pnl:.2%}")
        return False
    return True


def place_live_order(symbol: str, signal: str, quantity: float,
                     sl_price: float, tp_price: float) -> dict:
    """Sendet eine echte Market-Order mit Bracket (SL + TP)."""
    with open("config.json") as f:
        cfg = json.load(f)

    client = get_live_client()
    side   = OrderSide.BUY if signal.lower() == "buy" else OrderSide.SELL
    qty    = max(1, int(quantity))

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
        stop_loss=StopLossRequest(stop_price=round(sl_price, 2)),
    )
    order = client.submit_order(order_request)
    print(f"[Live] 💰 Order: {side.value.upper()} {qty}x {symbol} | "
          f"SL {sl_price:.2f} | TP {tp_price:.2f}")
    return {
        "order_id": str(order.id),
        "symbol":   symbol,
        "side":     side.value,
        "quantity": float(qty),
        "sl_price": sl_price,
        "tp_price": tp_price,
        "status":   str(order.status),
    }


def get_data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
    )


def get_account_value() -> float:
    return float(get_client().get_account().portfolio_value)


def get_buying_power() -> float:
    return float(get_client().get_account().buying_power)


def get_open_positions() -> list:
    return get_client().get_all_positions()


def get_sl_tp_map() -> dict:
    """
    Gibt ein Dict {symbol: {"sl": price, "tp": price}} zurück,
    indem alle offenen Stop/Limit-Orders (Bracket-Legs) geladen werden.
    """
    client = get_client()
    req    = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
    orders = client.get_orders(req)
    result = {}
    for o in orders:
        sym = o.symbol
        if sym not in result:
            result[sym] = {"sl": None, "tp": None}
        ot = str(o.order_type).lower()
        if "stop" in ot:
            price = float(o.stop_price or o.trail_price or 0) or None
            if price:
                result[sym]["sl"] = round(price, 2)
        elif "limit" in ot:
            price = float(o.limit_price or 0) or None
            if price:
                result[sym]["tp"] = round(price, 2)
    return result


def close_position(symbol: str):
    get_client().close_position(symbol)
    print(f"Position geschlossen: {symbol}")


def get_current_price(symbol: str) -> float:
    """Holt Marktpreis über IEX-Feed (kein SIP-Abo nötig)."""
    dc = get_data_client()
    try:
        quote = dc.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        )
        ask = float(quote[symbol].ask_price or 0)
        bid = float(quote[symbol].bid_price or 0)
        if ask > 0 and bid > 0:
            return round((ask + bid) / 2, 4)
    except Exception:
        pass
    trade = dc.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
    )
    return float(trade[symbol].price)


def get_bars(symbol: str, timeframe: TimeFrame, limit: int = 50) -> list:
    """Holt historische Bars für Multi-Timeframe und Volumen-Analyse."""
    dc = get_data_client()
    end   = datetime.now(pytz.UTC)
    start = end - timedelta(days=30)
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=limit,
        feed=DataFeed.IEX,
    )
    bars = dc.get_stock_bars(req)
    return bars[symbol] if symbol in bars else []


def check_market_hours() -> tuple[bool, str]:
    """
    Prüft ob der Markt offen ist und ob wir nicht in der ersten/letzten 30 Min handeln.
    Gibt (True, '') oder (False, Grund) zurück.
    """
    ny = pytz.timezone("America/New_York")
    now = datetime.now(ny)

    # Wochenende
    if now.weekday() >= 5:
        return False, "Wochenende - Markt geschlossen"

    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    with open("config.json") as f:
        cfg = json.load(f)
    t_start_h = cfg.get("trade_start_hour",   10)
    t_start_m = cfg.get("trade_start_minute", 30)
    t_end_h   = cfg.get("trade_end_hour",     14)
    t_end_m   = cfg.get("trade_end_minute",   30)

    avoid_start  = now.replace(hour=9,        minute=30,    second=0, microsecond=0)
    avoid_end    = now.replace(hour=t_start_h, minute=t_start_m, second=0, microsecond=0)
    avoid_start2 = now.replace(hour=t_end_h,  minute=t_end_m,   second=0, microsecond=0)

    if now < market_open or now > market_close:
        return False, "Markt geschlossen"
    if avoid_start <= now < avoid_end:
        return False, f"Vor Trading-Fenster (öffnet {t_start_h:02d}:{t_start_m:02d} NY)"
    if now >= avoid_start2:
        return False, f"Trading-Fenster geschlossen ({t_end_h:02d}:{t_end_m:02d} NY)"

    return True, ""


def check_volume(symbol: str) -> tuple[bool, str]:
    """
    Prüft ob das aktuelle Volumen überdurchschnittlich ist.
    Gibt (True, '') oder (False, Grund) zurück.
    """
    bars = get_bars(symbol, TimeFrame.Hour, limit=30)
    if len(bars) < 10:
        return True, ""  # Nicht genug Daten → nicht blockieren

    volumes   = [b.volume for b in bars]
    avg_vol   = sum(volumes[:-1]) / len(volumes[:-1])
    last_vol  = volumes[-1]

    with open("config.json") as f:
        threshold = json.load(f).get("volume_threshold", 0.8)

    if last_vol < avg_vol * threshold:
        return False, f"Volumen zu niedrig ({last_vol:,.0f} < {avg_vol * threshold:,.0f} Durchschnitt)"
    return True, ""


def check_multiframe_rsi(symbol: str, signal: str) -> tuple[bool, str]:
    """
    Prüft ob 1H und 4H RSI das Signal bestätigen.
    """
    def calc_rsi(bars, period=14):
        if len(bars) < period + 1:
            return 50
        closes = [b.close for b in bars]
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    bars_1h = get_bars(symbol, TimeFrame.Hour,  limit=30)
    bars_4h = get_bars(symbol, TimeFrame.Hour,  limit=80)

    rsi_1h = calc_rsi(bars_1h)
    rsi_4h = calc_rsi(bars_4h[::4]) if len(bars_4h) >= 20 else 50

    if signal == "buy":
        if rsi_1h > 50 and rsi_4h > 40:
            return True, ""
        return False, f"Multi-TF ablehnt BUY: 1H RSI={rsi_1h:.1f}, 4H RSI={rsi_4h:.1f}"
    else:
        if rsi_1h < 50 and rsi_4h < 60:
            return True, ""
        return False, f"Multi-TF ablehnt SELL: 1H RSI={rsi_1h:.1f}, 4H RSI={rsi_4h:.1f}"


def place_order(symbol: str, signal: str, quantity: float,
                sl_price: float, tp_price: float, trail_percent: float = None) -> dict:
    """
    Sendet eine Market Order mit Trailing Stop + Take Profit an Alpaca.
    trail_percent: z.B. 1.5 = SL bewegt sich 1.5% hinter dem Kurs
    """
    client = get_client()
    side   = OrderSide.BUY if signal.lower() == "buy" else OrderSide.SELL
    qty    = max(1, int(quantity))

    with open("config.json") as f:
        cfg = json.load(f)
    trail_pct = trail_percent or cfg["risk_management"].get("trailing_stop_percent", 2.5)

    # Fixer Stop beim Einstieg — Trailing wird separat via update_trailing_stops() nachgezogen
    stop_loss_req = StopLossRequest(
        stop_price=round(sl_price, 2),
    )

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
        stop_loss=stop_loss_req,
    )

    order = client.submit_order(order_request)
    print(f"Order: {side.value} {qty} {symbol} | Trail: {trail_pct}% | TP: {tp_price}")

    return {
        "order_id":      str(order.id),
        "symbol":        symbol,
        "side":          side.value,
        "quantity":      float(qty),
        "sl_price":      sl_price,
        "tp_price":      tp_price,
        "trail_percent": trail_pct,
        "status":        str(order.status),
    }


def check_daily_loss_limit() -> bool:
    with open("config.json") as f:
        config = json.load(f)
    max_loss_pct = config["risk_management"]["max_daily_loss_percent"] / 100
    account      = get_client().get_account()
    equity       = float(account.equity)
    last_equity  = float(account.last_equity)
    daily_pnl    = (equity - last_equity) / last_equity
    if daily_pnl <= -max_loss_pct:
        print(f"Tägliches Verlustlimit erreicht! PnL: {daily_pnl:.2%}")
        return False
    return True


def get_closed_trades_today() -> list:
    """Holt heute geschlossene Orders für Win-Rate Berechnung."""
    client = get_client()
    ny     = pytz.timezone("America/New_York")
    today  = datetime.now(ny).replace(hour=0, minute=0, second=0, microsecond=0)
    req    = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=today, limit=50)
    orders = client.get_orders(req)
    result = []
    for o in orders:
        if str(o.status) == "OrderStatus.filled":
            filled_price = float(o.filled_avg_price or 0)
            result.append({
                "symbol":   o.symbol,
                "side":     str(o.side.value),
                "qty":      float(o.qty or 0),
                "price":    filled_price,
                "order_id": str(o.id),
            })
    return result
