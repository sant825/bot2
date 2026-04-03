"""
Multi-Symbol XGBoost Training
Trainiert ein gemeinsames Modell auf AAPL, GLD und SPY
"""

import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SYMBOLS = ["AAPL", "GLD", "SPY"]
START   = "2018-01-01"
END     = "2026-01-01"


def calculate_features(df):
    df = df.copy()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["EMA50"]  = df["Close"].ewm(span=50,  adjust=False).mean()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    # Abstand zum EMA als Feature
    df["EMA200_dist"] = (df["Close"] - df["EMA200"]) / df["EMA200"] * 100
    df["EMA50_dist"]  = (df["Close"] - df["EMA50"])  / df["EMA50"]  * 100

    # Ziel: Steigt Preis in 5 Tagen?
    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
    return df.dropna()


all_frames = []

for symbol in SYMBOLS:
    print(f"Lade {symbol}...")
    raw = yf.download(symbol, start=START, end=END, auto_adjust=True)
    # MultiIndex Spalten flachdrücken (yfinance Eigenheit)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df  = calculate_features(raw)
    df["Symbol"] = symbol
    all_frames.append(df)

combined = pd.concat(all_frames)

features = ["RSI", "EMA200_dist", "EMA50_dist", "ATR"]
X = combined[features]
y = combined["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrainiere XGBoost auf {len(X_train)} Datenpunkten ({', '.join(SYMBOLS)})...")
model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

preds    = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Modell-Genauigkeit: {accuracy * 100:.2f}%")

model.save_model("trading_model.json")
print("Modell gespeichert: trading_model.json")
print(f"Trainiert auf: {', '.join(SYMBOLS)}")
