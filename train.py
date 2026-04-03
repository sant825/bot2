import yfinance as yf
import pandas as pd
import xgboost as xgb

# 1. Daten laden
print("Lade Kursdaten von Yahoo Finance...")
data = yf.download("AAPL", start="2022-01-01", end="2026-01-01")

# 2. Features berechnen
def calculate_features(df):
    df = df.copy()
    # EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Ziel: Steigt der Preis in 5 Tagen?
    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    return df.dropna()

df = calculate_features(data)

# 3. Modell trainieren
X = df[['RSI', 'EMA200']]
y = df['Target']

print("Trainiere die KI (XGBoost)...")
model = xgb.XGBClassifier()
model.fit(X, y)

# 4. DAS MODELL SPEICHERN (WICHTIG!)
model.save_model("trading_model.json")
print("ERFOLG: 'trading_model.json' wurde erstellt und ist bereit!")