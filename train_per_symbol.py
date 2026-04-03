"""
Trainiert ein separates XGBoost-Modell pro Symbol.
Speichert: model_AAPL.json, model_GLD.json, model_SPY.json
"""

import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SYMBOLS = {"AAPL": "2018-01-01", "GLD": "2018-01-01", "SPY": "2018-01-01"}


def calculate_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["EMA200"]     = df["Close"].ewm(span=200, adjust=False).mean()
    df["EMA50"]      = df["Close"].ewm(span=50,  adjust=False).mean()
    delta            = df["Close"].diff()
    gain             = delta.where(delta > 0, 0).rolling(14).mean()
    loss             = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"]        = 100 - (100 / (1 + gain / loss))
    df["ATR"]        = (df["High"] - df["Low"]).rolling(14).mean()
    df["EMA200_dist"] = (df["Close"] - df["EMA200"]) / df["EMA200"] * 100
    df["EMA50_dist"]  = (df["Close"] - df["EMA50"])  / df["EMA50"]  * 100
    df["Vol_ratio"]   = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Target"]      = (df["Close"].shift(-5) > df["Close"]).astype(int)
    return df.dropna()


FEATURES = ["RSI", "EMA200_dist", "EMA50_dist", "ATR", "Vol_ratio"]

for symbol, start in SYMBOLS.items():
    print(f"\n--- {symbol} ---")
    raw = yf.download(symbol, start=start, end="2026-01-01", auto_adjust=True)
    df  = calculate_features(raw)

    X = df[FEATURES]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Genauigkeit: {acc * 100:.2f}%")

    model.save_model(f"model_{symbol}.json")
    print(f"Gespeichert: model_{symbol}.json")

print("\nAlle Modelle trainiert!")
