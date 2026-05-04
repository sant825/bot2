"""
Risikomanagement: Berechnet Positionsgröße, Stop Loss und Take Profit
"""

import json


def load_config():
    with open("config.json") as f:
        return json.load(f)


def calculate_trade(entry_price: float, signal: str, account_value: float,
                    atr: float = None, score: float = None,
                    risk_pct_override: float = None) -> dict:
    """
    Berechnet SL, TP und Positionsgröße basierend auf Risikomanagement-Regeln.

    Args:
        entry_price:   Einstiegspreis des Assets
        signal:        'buy' oder 'sell'
        account_value: Aktueller Kontostand in USD
        atr:           Average True Range (optional, für dynamischen SL)
        score:         Screener-Score 0-100 (optional, skaliert Positionsgröße)

    Returns:
        dict mit sl_price, tp_price, quantity, risk_amount
    """
    config   = load_config()["risk_management"]
    base_pct = config["risk_per_trade_percent"] / 100
    rr_ratio = config["risk_reward_ratio"]

    # Explizite Überschreibung hat höchste Priorität (z.B. Live-Trading)
    if risk_pct_override is not None:
        risk_pct = risk_pct_override
    # Score-basiertes Risiko: Score 50 → 0.5x, 75 → 1.0x, 100 → 1.5x Basisrisiko
    elif score is not None and score >= 50:
        multiplier = 0.5 + (score - 50) / 50.0   # linear 0.5..1.5
        multiplier = min(max(multiplier, 0.5), 1.5)
        risk_pct   = base_pct * multiplier
    else:
        risk_pct = base_pct

    risk_amount = account_value * risk_pct

    # SL-Abstand: 1% vom Einstiegspreis (oder ATR falls übergeben)
    if atr:
        sl_distance = atr * 2.5
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
        "entry_price":  round(entry_price, 4),
        "sl_price":     round(sl_price, 4),
        "tp_price":     round(tp_price, 4),
        "quantity":     quantity,
        "risk_amount":  round(risk_amount, 2),
        "risk_percent": round(risk_pct * 100, 3),
        "rr_ratio":     rr_ratio,
        "signal":       signal,
        "score":        score,
    }
