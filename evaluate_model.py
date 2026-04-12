"""
Evaluate a trained model by generating candidate decks for a list of commanders and running sims to compare model picks vs baseline.
Usage:
    python evaluate_model.py --model model.joblib --commanders commanders.txt --budget 100 --sims 500 --candidates 50 --topk 5
"""

import argparse
import joblib
import os
from optimizer_ml_v2 import expand_deck_lines, collapse_deck_list, mutate_cards
from deck_builder import build_deck, get_edhrec_synergy
import tcg_simulator as sim
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model.joblib')
parser.add_argument('--commanders', required=True)
parser.add_argument('--budget', type=float, required=True)
parser.add_argument('--sims', type=int, default=500)
parser.add_argument('--candidates', type=int, default=100)
parser.add_argument('--topk', type=int, default=5)
args = parser.parse_args()

m = joblib.load(args.model)
model = m.get('model')
vocab = m.get('vocab')

for cmdr in open(args.commanders):
    cmdr = cmdr.strip()
    if not cmdr:
        continue
    print('\nEvaluating model for', cmdr)
    baseline_lines, card_db = build_deck(cmdr, total_budget=args.budget, print_list=False)
    commander = baseline_lines[0].split(' ',1)[1].replace('*CMDR*','').strip()
    base_cards = expand_deck_lines(baseline_lines)
    if commander in base_cards:
        base_cards.remove(commander)
        base_cards.insert(0, commander)

    # generate candidates
    pool_raw = get_edhrec_synergy(cmdr)
    pool = [item['name'] for item in pool_raw if item['name'] in card_db]
    if len(pool) < 200:
        pool = list(card_db.keys())
    cands = []
    while len(cands) < args.candidates:
        child = mutate_cards(base_cards, pool, card_db, args.budget, commander)
        if child not in cands:
            cands.append(child)

    # predict
    def predict(cand):
        from optimizer_ml_v2 import predict_deck_with_model
        return predict_deck_with_model(cand, model, vocab)

    preds = sorted([(predict(c), c) for c in cands], key=lambda x: x[0], reverse=True)
    top = [c for _, c in preds[:args.topk]]

    # evaluate top candidates
    for i, cand in enumerate(top):
        deck_lines = collapse_deck_list(cand, commander)
        deck_sim = sim.parse_deck_list(deck_lines, card_db)
        decks = [deepcopy(deck_sim) for _ in range(4)]
        res = sim.run_multiplayer_tournament(decks, n_games=args.sims)
        print(f"Candidate {i+1}: winrate={res.get('Player1',0)/args.sims:.3f}")

    # baseline
    deck_lines = baseline_lines
    deck_sim = sim.parse_deck_list(deck_lines, card_db)
    decks = [deepcopy(deck_sim) for _ in range(4)]
    res = sim.run_multiplayer_tournament(decks, n_games=args.sims)
    print('Baseline winrate:', res.get('Player1',0)/args.sims)
