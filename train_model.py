import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Daten laden (Beispiel: Apple)
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# 2. Features berechnen (RSI & EMA)
def calculate_features(df):
    # EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Ziel-Variable: Preis in 5 Tagen > Preis heute? (1 = Profit, 0 = Verlust)
    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    return df.dropna()

df = calculate_features(data)

# 3. Vorbereitung für XGBoost
# Wir nutzen RSI und den Abstand zum EMA als Entscheidungsgrundlage
X = df[['RSI', 'EMA200']] 
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modell trainieren
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

# 5. Vorhersage testen
preds = model.predict(X_test)
print(f"Genauigkeit des Modells: {accuracy_score(y_test, preds) * 100:.2f}%")

# Beispiel-Abfrage für einen neuen Trade:
# Angenommen RSI ist 35 und Kurs ist 150 (EMA ist 145)
new_data = pd.DataFrame([[35, 145]], columns=['RSI', 'EMA200'])
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("XGBoost sagt: WAHRSCHEINLICH PROFITABEL - KAUFEN")
else:
    print("XGBoost sagt: ZU RISKANT - NICHT KAUFEN")


model.save_model("trading_model.json")
print("Modell erfolgreich gespeichert!")

