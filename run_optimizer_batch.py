"""
Non-interactive entrypoint to run a long optimizer run.
Usage:
    python run_optimizer_batch.py --commander "Atraxa, praetor's voice" --budget 100 --sims 2000 --pop 20 --gens 50

Outputs training_data.csv in the repository root as optimizer collects examples.
"""

import argparse
from optimizer_ml_v2 import run_search

parser = argparse.ArgumentParser(description='Run optimizer non-interactively')
parser.add_argument('--commander', required=True)
parser.add_argument('--budget', type=float, required=True)
parser.add_argument('--sims', type=int, default=2000)
parser.add_argument('--pop', type=int, default=20)
parser.add_argument('--gens', type=int, default=50)
parser.add_argument('--training-csv', default='training_data.csv', help='Path to write per-run training CSV')
parser.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducibility')

args = parser.parse_args()

if __name__ == '__main__':
    print(f"Starting optimizer: {args.commander}, budget=${args.budget}, sims={args.sims}, pop={args.pop}, gens={args.gens}")
    run_search(args.commander, args.budget, n_games=args.sims, population_size=args.pop, generations=args.gens, training_csv=args.training_csv, seed=args.seed)
