"""
Alpaca Broker Wrapper - Paper Trading & Live Trading
Einfacher Wechsel zwischen Paper und Live durch config.json
"""

import os
import json
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

load_dotenv()


def get_client() -> TradingClient:
    with open("config.json") as f:
        config = json.load(f)
    paper = config["alpaca"]["paper_trading"]
    return TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=paper,
    )


def get_account_value() -> float:
    client = get_client()
    account = client.get_account()
    return float(account.portfolio_value)


def get_open_positions() -> list:
    client = get_client()
    return client.get_all_positions()


def get_current_price(symbol: str) -> float:
    """Holt den aktuellen Marktpreis von Alpaca. Fallback auf letzten Trade-Preis."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    data_client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
    )
    # Erst Quote versuchen (Markt offen)
    quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data_client.get_stock_latest_quote(quote_req)
    ask = float(quote[symbol].ask_price or 0)
    bid = float(quote[symbol].bid_price or 0)
    if ask > 0 and bid > 0:
        return round((ask + bid) / 2, 4)
    # Fallback: letzter Trade-Preis (auch außerhalb Marktzeiten)
    trade_req = StockLatestTradeRequest(symbol_or_symbols=symbol)
    trade = data_client.get_stock_latest_trade(trade_req)
    return float(trade[symbol].price)


def close_position(symbol: str):
    client = get_client()
    client.close_position(symbol)
    print(f"Position geschlossen: {symbol}")


def get_buying_power() -> float:
    client = get_client()
    account = client.get_account()
    return float(account.buying_power)


def place_order(symbol: str, signal: str, quantity: float, sl_price: float, tp_price: float) -> dict:
    """
    Sendet eine Market Order mit Bracket (SL + TP) an Alpaca.

    Args:
        symbol:   z.B. 'AAPL' oder 'EURUSD'
        signal:   'buy' oder 'sell'
        quantity: Anzahl Aktien/Units
        sl_price: Stop Loss Preis
        tp_price: Take Profit Preis

    Returns:
        Order-Details als dict
    """
    client = get_client()
    side = OrderSide.BUY if signal.lower() == "buy" else OrderSide.SELL

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=max(1, int(quantity)),
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
        stop_loss=StopLossRequest(stop_price=round(sl_price, 2)),
    )

    order = client.submit_order(order_request)
    print(f"Order gesendet: {side.value} {quantity} {symbol} | SL: {sl_price} | TP: {tp_price}")

    return {
        "order_id": str(order.id),
        "symbol": symbol,
        "side": side.value,
        "quantity": float(quantity),
        "sl_price": sl_price,
        "tp_price": tp_price,
        "status": str(order.status),
    }


def check_daily_loss_limit() -> bool:
    """
    Gibt True zurück wenn das tägliche Verlustlimit noch NICHT erreicht ist.
    """
    with open("config.json") as f:
        config = json.load(f)
    max_loss_pct = config["risk_management"]["max_daily_loss_percent"] / 100

    client = get_client()
    account = client.get_account()
    equity = float(account.equity)
    last_equity = float(account.last_equity)

    daily_pnl_pct = (equity - last_equity) / last_equity
    if daily_pnl_pct <= -max_loss_pct:
        print(f"Tägliches Verlustlimit erreicht! PnL: {daily_pnl_pct:.2%}")
        return False
    return True
