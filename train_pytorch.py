"""
Train a simple PyTorch MLP on the training_data.csv features.
This is a best-effort script; install torch and adapt hyperparams as needed.

Usage:
    pip install torch torchvision
    python train_pytorch.py --data training_data.csv --out model.pt
"""

import argparse
import os
import csv
import joblib
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    print('PyTorch not installed. pip install torch')
    raise

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='training_data.csv')
parser.add_argument('--out', default='model.pt')
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

if not os.path.exists(args.data):
    print('No data found at', args.data)
    exit(1)

# Very simple loader: read csv, build bag-of-cards vocab and engineered features
rows = []
with open(args.data, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

from collections import Counter
all_cards = Counter()
for r in rows:
    cards = r.get('cards_serialized','').split(';') if r.get('cards_serialized') else []
    all_cards.update(cards)
TOP_N = 2000
vocab = [c for c,_ in all_cards.most_common(TOP_N)]
index = {c:i for i,c in enumerate(vocab)}

X = []
y = []
for r in rows:
    vec = [0]*len(vocab)
    cards = r.get('cards_serialized','').split(';') if r.get('cards_serialized') else []
    for c in cards:
        if c in index:
            vec[index[c]] += 1
    # engineered features
    engineered_keys = ['avg_mana','avg_price','num_creatures','num_spells','num_lands'] + [f'mana_bin_{i}' for i in range(8)] + [f'color_{c}' for c in ['W','U','B','R','G']]
    for k in engineered_keys:
        vec.append(float(r.get(k,0)))
    X.append(vec)
    y.append(float(r.get('winrate',0)))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# simple MLP
input_dim = X.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y).unsqueeze(1)

for epoch in range(args.epochs):
    model.train()
    opt.zero_grad()
    preds = model(X_t)
    loss = loss_fn(preds, y_t)
    loss.backward()
    opt.step()
    print(f'Epoch {epoch+1}/{args.epochs} loss={loss.item():.6f}')

# save model and vocab
torch.save({'model_state': model.state_dict(), 'vocab': vocab}, args.out)
print('Saved PyTorch model to', args.out)
