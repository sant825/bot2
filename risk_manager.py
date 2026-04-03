"""
Risikomanagement: Berechnet Positionsgröße, Stop Loss und Take Profit
"""

import json


def load_config():
    with open("config.json") as f:
        return json.load(f)


def calculate_trade(entry_price: float, signal: str, account_value: float, atr: float = None) -> dict:
    """
    Berechnet SL, TP und Positionsgröße basierend auf Risikomanagement-Regeln.

    Args:
        entry_price: Einstiegspreis des Assets
        signal:      'buy' oder 'sell'
        account_value: Aktueller Kontostand in USD
        atr:         Average True Range (optional, für dynamischen SL)

    Returns:
        dict mit sl_price, tp_price, quantity, risk_amount
    """
    config = load_config()["risk_management"]
    risk_pct = config["risk_per_trade_percent"] / 100
    rr_ratio = config["risk_reward_ratio"]

    risk_amount = account_value * risk_pct

    # SL-Abstand: 1% vom Einstiegspreis (oder ATR falls übergeben)
    if atr:
        sl_distance = atr * 1.5
    else:
        sl_distance = entry_price * 0.01  # 1% default

    if signal.lower() == "buy":
        sl_price = entry_price - sl_distance
        tp_price = entry_price + (sl_distance * rr_ratio)
    elif signal.lower() == "sell":
        sl_price = entry_price + sl_distance
        tp_price = entry_price - (sl_distance * rr_ratio)
    else:
        raise ValueError(f"Unbekanntes Signal: {signal}. Erwartet 'buy' oder 'sell'.")

    quantity = risk_amount / sl_distance
    quantity = max(1, round(quantity, 4))

    return {
        "entry_price": round(entry_price, 4),
        "sl_price": round(sl_price, 4),
        "tp_price": round(tp_price, 4),
        "quantity": quantity,
        "risk_amount": round(risk_amount, 2),
        "risk_percent": config["risk_per_trade_percent"],
        "rr_ratio": rr_ratio,
        "signal": signal,
    }
