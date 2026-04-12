"""
Run optimizer sequentially for multiple commanders listed in a file and save best decks.
Usage:
    python multi_commander_runner.py --file commanders.txt --budget 100 --sims 2000 --pop 20 --gens 50 --out results

Outputs:
 - results/summary.csv (commander, price, winrate, deck_file)
 - results/<safe_commander_name>.deck (decklist in moxfield format)
"""

import argparse
import os
import csv
import re
from optimizer_ml_v2 import run_search


def safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]', '_', name).strip('_')[:200]


def main():
    parser = argparse.ArgumentParser(description='Run optimizer for multiple commanders')
    parser.add_argument('--file', required=True, help='Path to commanders file (one commander per line)')
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--sims', type=int, default=2000)
    parser.add_argument('--pop', type=int, default=20)
    parser.add_argument('--gens', type=int, default=50)
    parser.add_argument('--out', default='results')
    parser.add_argument('--training-csv-template', default='training_{safe}.csv', help='Template for per-commander training CSV; {safe} replaced with safe name')
    parser.add_argument('--seed-base', type=int, default=1000, help='Base seed; per-commander seed = seed_base + index')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, 'summary.csv')
    with open(args.file, encoding='utf-8') as f, open(summary_path, 'w', newline='', encoding='utf-8') as summary_f:
        reader = (line.strip() for line in f if line.strip())
        writer = csv.writer(summary_f)
        writer.writerow(['commander', 'price', 'winrate', 'deck_file', 'training_csv'])

        for idx, cmdr in enumerate(reader):
            print(f"\n=== Running optimizer for: {cmdr} ===")
            safe = safe_name(cmdr)
            training_csv = args.training_csv_template.format(safe=safe)
            seed = args.seed_base + idx
            res = run_search(cmdr, args.budget, n_games=args.sims, population_size=args.pop, generations=args.gens, training_csv=training_csv, seed=seed)
            # res is expected to be a dict
            comm = res.get('commander', cmdr)
            deck = res.get('deck') or []
            price = res.get('price', 0.0)
            winrate = res.get('winrate', 0.0)

            deck_file = os.path.join(args.out, f"{safe}.deck")
            if deck:
                with open(deck_file, 'w', encoding='utf-8') as df:
                    df.write('\n'.join(deck))
            else:
                deck_file = ''

            writer.writerow([comm, f"{price:.2f}", f"{winrate:.4f}", deck_file, training_csv])
            summary_f.flush()

    print(f"\nAll done. Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
