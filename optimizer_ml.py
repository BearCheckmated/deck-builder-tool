"""
Optimizer with active learning loop: uses a model to pre-score candidates, simulates top candidates, retrains model periodically,
and guides genetic search to improve decks over time.
"""
from __future__ import annotations
import random
import time
import os
import csv
import re
from copy import deepcopy
from typing import List, Dict, Tuple

from deck_builder import build_deck, get_edhrec_synergy
import tcg_simulator as sim

# Try to import sklearn; if unavailable, the optimizer will still run but without model retraining
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def expand_deck_lines(deck_lines: List[str]) -> List[str]:
    cards = []
    for line in deck_lines:
        parts = line.split(" ", 1)
        try:
            count = int(parts[0])
            name = parts[1].replace("*CMDR*", "").strip()
        except Exception:
            continue
        for _ in range(count):
            cards.append(name)
    return cards


def collapse_deck_list(cards: List[str], commander: str) -> List[str]:
    out = []
    counts: Dict[str, int] = {}
    for c in cards:
        counts[c] = counts.get(c, 0) + 1
    if commander in counts:
        counts[commander] = 1
    out.append(f"1 {commander} *CMDR*")
    for name, cnt in counts.items():
        if name == commander:
            continue
        out.append(f"{cnt} {name}")
    return out


def deck_price(deck_lines: List[str], card_db: Dict[str, dict]) -> float:
    total = 0.0
    for line in deck_lines:
        parts = line.split(" ", 1)
        try:
            cnt = int(parts[0])
            name = parts[1].replace("*CMDR*", "").strip()
        except Exception:
            continue
        entry = card_db.get(name, {})
        price = float(entry.get("price") or 0)
        total += cnt * price
    return total


def mutate_cards(cards: List[str], pool: List[str], card_db: Dict[str, dict], budget: float, commander: str) -> List[str]:
    new_cards = cards[:]  # list of names
    mutable_idx = [i for i, c in enumerate(new_cards) if c != commander and c.lower() not in ("forest","island","swamp","mountain","plains","wastes")]
    if not mutable_idx:
        return new_cards
    n_swaps = random.randint(1, min(3, len(mutable_idx)))
    for _ in range(n_swaps):
        idx = random.choice(mutable_idx)
        candidates = [p for p in pool if p not in new_cards and p != commander]
        if not candidates:
            break
        choice = random.choice(candidates)
        old = new_cards[idx]
        new_cards[idx] = choice
        tmp_lines = collapse_deck_list(new_cards, commander)
        price = deck_price(tmp_lines, card_db)
        if price > budget:
            cheaper = sorted(candidates, key=lambda n: float(card_db.get(n, {}).get('price') or 0))
            replaced = False
            for cand in cheaper:
                new_cards[idx] = cand
                if deck_price(collapse_deck_list(new_cards, commander), card_db) <= budget:
                    replaced = True
                    break
            if not replaced:
                new_cards[idx] = old
    return new_cards


def initial_population(baseline_lines: List[str], card_db: Dict[str, dict], pool: List[str], commander: str, population_size: int, budget: float) -> List[List[str]]:
    base_cards = expand_deck_lines(baseline_lines)
    pop = []
    pop.append(base_cards)
    while len(pop) < population_size:
        candidate = mutate_cards(base_cards, pool, card_db, budget, commander)
        pop.append(candidate)
    return pop


# ----------------- Model helpers -----------------

def load_model(path: str = 'model.joblib') -> Tuple[object, List[str]]:
    if not SKLEARN_AVAILABLE:
        return None, None
    if not os.path.exists(path):
        return None, None
    try:
        data = joblib.load(path)
        return data.get('model'), data.get('vocab')
    except Exception:
        return None, None


def save_model(model, vocab: List[str], path: str = 'model.joblib'):
    if not SKLEARN_AVAILABLE:
        return
    joblib.dump({'model': model, 'vocab': vocab}, path)


def retrain_model_from_csv(path: str = 'training_data.csv', top_n_cards: int = 2000) -> Tuple[object, List[str]]:
    if not SKLEARN_AVAILABLE:
        print('[MODEL] sklearn not available; skipping retrain')
        return None, None
    if not os.path.exists(path):
        print('[MODEL] No training data found.')
        return None, None
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None, None
    from collections import Counter
    all_cards = Counter()
    for r in rows:
        cards = r['cards_serialized'].split(';') if r['cards_serialized'] else []
        all_cards.update(cards)
    vocab = [c for c, _ in all_cards.most_common(top_n_cards)]
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
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)
    save_model(model, vocab)
    print('[MODEL] Retrained and saved model.joblib')
    return model, vocab


def predict_deck_with_model(deck_cards: List[str], model, vocab: List[str]) -> float:
    if model is None or vocab is None:
        return 0.0
    index = {c: i for i, c in enumerate(vocab)}
    vec = [0] * len(vocab)
    for c in deck_cards:
        if c in index:
            vec[index[c]] += 1
    try:
        return float(model.predict([vec])[0])
    except Exception:
        return 0.0


# ----------------- Simulation evaluation -----------------

def evaluate_candidate_sim(deck_cards: List[str], card_db: Dict[str, dict], n_games: int, commander: str) -> float:
    deck_lines = collapse_deck_list(deck_cards, commander)
    deck_cards_sim = sim.parse_deck_list(deck_lines, card_db)
    decks = [deepcopy(deck_cards_sim) for _ in range(4)]
    results = sim.run_multiplayer_tournament(decks, n_games=n_games)
    wins = results.get('Player1', 0)
    winrate = wins / n_games
    # append to training data
    try:
        write_header = not os.path.exists('training_data.csv')
        with open('training_data.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['commander', 'cards_serialized', 'winrate'])
            writer.writerow([commander, ';'.join(deck_cards), winrate])
    except Exception as e:
        print(f"[WARN] Could not write training data: {e}")
    return winrate


# ----------------- Active-learning genetic search -----------------

def run_search(cmdr_name: str, budget: float, n_games: int = 500, population_size: int = 6, generations: int = 5,
               candidate_pool_size: int = 200, fast_sims: int = 50, top_k_full: int = 5, retrain_every: int = 2) -> dict:
    print(f"[OPT-ML] Building baseline deck for {cmdr_name} with budget ${budget:.2f}...")
    baseline_lines, card_db = build_deck(cmdr_name, total_budget=budget, print_list=False)
    if not baseline_lines:
        print("[OPT-ML] Baseline build failed.")
        return {"commander": cmdr_name, "deck": None, "price": 0.0, "winrate": 0.0}

    raw = get_edhrec_synergy(cmdr_name)
    pool = [item['name'] for item in raw if item['name'] in card_db]
    if len(pool) < 200:
        pool = list(card_db.keys())

    commander = baseline_lines[0].split(" ", 1)[1].replace("*CMDR*", "").strip()
    base_cards = expand_deck_lines(baseline_lines)
    if commander in base_cards:
        base_cards.remove(commander)
        base_cards.insert(0, commander)

    population = initial_population(baseline_lines, card_db, pool, commander, population_size, budget)

    model, vocab = load_model()
    if model is not None:
        print('[OPT-ML] Loaded existing model; using it to pre-score candidates')

    best = None
    best_score = -1.0

    # Track how many new labeled examples were added since last retrain
    new_labels_since_retrain = 0

    for gen in range(1, generations + 1):
        print(f"[OPT-ML] Generation {gen}/{generations}")

        # Expand candidate pool by mutating current population
        candidate_pool = []
        while len(candidate_pool) < candidate_pool_size:
            parent = random.choice(population)
            child = mutate_cards(parent, pool, card_db, budget, commander)
            if child not in candidate_pool:
                candidate_pool.append(child)

        # If model available, predict fast scores
        model_scores = {}
        if model and vocab:
            for cand in candidate_pool:
                model_scores[tuple(cand)] = predict_deck_with_model(cand, model, vocab)
        else:
            for cand in candidate_pool:
                model_scores[tuple(cand)] = random.random()

        # Select top candidates for fast sims
        sorted_candidates = sorted(candidate_pool, key=lambda c: model_scores[tuple(c)], reverse=True)
        fast_selected = sorted_candidates[: min(len(sorted_candidates), population_size * 4)]

        # Run fast sims to get noisy labels
        fast_results = {}
        print(f"[OPT-ML] Running fast sims for {len(fast_selected)} candidates (sims={fast_sims})...")
        for cand in fast_selected:
            win = evaluate_candidate_sim(cand, card_db, n_games=fast_sims, commander=commander)
            fast_results[tuple(cand)] = win
            new_labels_since_retrain += 1

        # Choose top candidates from fast_results for full sims
        candidates_for_full = sorted(fast_results.items(), key=lambda x: x[1], reverse=True)[:top_k_full]
        print(f"[OPT-ML] Running full sims for top {len(candidates_for_full)} candidates (sims={n_games})...")
        full_results = {}
        for tup, _ in candidates_for_full:
            cand = list(tup)
            win = evaluate_candidate_sim(cand, card_db, n_games=n_games, commander=commander)
            full_results[tuple(cand)] = win
            new_labels_since_retrain += 1
            if win > best_score:
                best_score = win
                best = deepcopy(cand)

        # Combine results for selection: prefer full_results, then fast_results, else model score
        scored = []
        for cand in population:
            key = tuple(cand)
            score = full_results.get(key) or fast_results.get(key) or model_scores.get(key) or 0.0
            scored.append((score, cand))

        # Add newly evaluated candidates to selection pool
        for key, val in list(full_results.items()) + list(fast_results.items()):
            scored.append((val, list(key)))

        # Select survivors
        scored.sort(key=lambda x: x[0], reverse=True)
        survivors = [ind for (_, ind) in scored[: max(1, len(scored)//2)]]

        # Reproduce
        new_pop = []
        while len(new_pop) < population_size:
            parent = random.choice(survivors)
            child = mutate_cards(parent, pool, card_db, budget, commander)
            if commander not in child:
                child[0] = commander
            new_pop.append(child)
        population = new_pop

        # Retrain model periodically if enough new labels were added
        if SKLEARN_AVAILABLE and new_labels_since_retrain >= 20 and (gen % retrain_every == 0):
            print('[OPT-ML] Retraining model from training_data.csv...')
            model, vocab = retrain_model_from_csv()
            new_labels_since_retrain = 0

    if best:
        best_lines = collapse_deck_list(best, commander)
        price = deck_price(best_lines, card_db)
        print('\n[OPT-ML] Best deck found:')
        for line in best_lines:
            print(line)
        print(f'Estimated price: ${price:.2f}')
        print(f'Estimated winrate (during search): {best_score:.3f}')
        return {"commander": cmdr_name, "deck": best_lines, "price": price, "winrate": best_score}
    else:
        print('[OPT-ML] No candidate found.')
        return {"commander": cmdr_name, "deck": None, "price": 0.0, "winrate": 0.0}
