"""
Simple monitor: prints number of training rows, model existence, and latest best decks.
Usage: python monitor_progress.py
"""
import os
import time
import csv

TRAINING = 'training_data.csv'
MODEL = 'model.joblib'

def tail_latest_decks(n=5, results_dir='results'):
    if not os.path.isdir(results_dir):
        return []
    files = [f for f in os.listdir(results_dir) if f.endswith('.deck')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir,f)), reverse=True)
    out = []
    for f in files[:n]:
        path = os.path.join(results_dir, f)
        with open(path, encoding='utf-8') as fh:
            lines = fh.read().splitlines()
            out.append((f, lines[:10]))
    return out

if __name__ == '__main__':
    print('Monitoring... (Ctrl-C to exit)')
    try:
        while True:
            rows = 0
            if os.path.exists(TRAINING):
                with open(TRAINING, newline='', encoding='utf-8') as f:
                    rows = sum(1 for _ in f) - 1
            print(f'Training rows: {rows}')
            print('Model exists:', os.path.exists(MODEL))
            decks = tail_latest_decks()
            for name, lines in decks:
                print('Deck:', name)
                for ln in lines:
                    print('  ', ln)
            time.sleep(30)
    except KeyboardInterrupt:
        pass
