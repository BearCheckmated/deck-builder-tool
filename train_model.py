"""
Simple training script to fit a regressor predicting winrate from bag-of-cards features.
- Uses scikit-learn if available.
- Input: training_data.csv produced by optimizer.py
- Output: saves a trained model to model.joblib

Run on your server after collecting enough rows (thousands recommended):
    pip install scikit-learn joblib
    python train_model.py
"""

import os
import csv
from collections import Counter

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import joblib
except Exception:
    print("scikit-learn or joblib not installed. Please pip install scikit-learn joblib")
    raise

DATA_FILE = 'training_data.csv'
MODEL_FILE = 'model.joblib'

if not os.path.exists(DATA_FILE):
    print(f"No {DATA_FILE} found. Run optimizer to generate training examples first.")
    exit(1)

# Build vocabulary of card names
rows = []
with open(DATA_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    print("No training rows found.")
    exit(1)

all_cards = Counter()
for r in rows:
    cards = r['cards_serialized'].split(';') if r['cards_serialized'] else []
    all_cards.update(cards)

# Keep top N cards to limit dimensionality
TOP_N = 2000
vocab = [c for c, _ in all_cards.most_common(TOP_N)]
index = {c: i for i, c in enumerate(vocab)}

X = []
y = []
for r in rows:
    vec = [0] * len(vocab)
    cards = r['cards_serialized'].split(';') if r['cards_serialized'] else []
    for c in cards:
        if c in index:
            vec[index[c]] += 1
    X.append(vec)
    y.append(float(r['winrate']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training RandomForestRegressor on", len(X_train), "examples...")
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))

joblib.dump({'model': model, 'vocab': vocab}, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
