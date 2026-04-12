"""
Improved optimizer with feature-enriched training data, per-run training files, seed support,
and combined bag-of-cards + engineered features for the model.
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


def extract_deck_features(deck_cards: List[str], card_db: Dict[str, dict]) -> Dict[str, float]:
    # features: avg_mana, mana_curve bins 0..6+, num_creatures, num_spells, num_lands, avg_price, color_counts WUBRG
    mana_bins = [0] * 8  # 0..6, 7 means 6+
    num_creatures = 0
    num_spells = 0
    num_lands = 0
    total_mana = 0.0
    total_price = 0.0
    color_counts = {c: 0 for c in ['W','U','B','R','G']}
    for name in deck_cards:
        raw = card_db.get(name) or {}
        mc = raw.get('mana_cost', '')
        # approximate mana by counting symbols like {1} {G} etc. fallback to CMC if present
        mana = 0
        if isinstance(mc, str) and mc:
            mana = len([c for c in mc.split('}') if c.strip('{').strip()])
        else:
            try:
                mana = int(raw.get('cmc') or 0)
            except Exception:
                mana = 0
        total_mana += mana
        b = mana if mana <= 6 else 7
        mana_bins[b] += 1

        tline = (raw.get('type_line') or '').lower()
        if 'land' in tline:
            num_lands += 1
        elif 'creature' in tline:
            num_creatures += 1
        elif 'instant' in tline or 'sorcery' in tline or 'enchant' in tline or 'artifact' in tline:
            num_spells += 1
        price = float(raw.get('price') or 0)
        total_price += price
        for i, ch in enumerate(['W','U','B','R','G']):
            if ch in (raw.get('identity') or []):
                color_counts[ch] += 1
    n = max(1, len(deck_cards))
    features = {
        'avg_mana': total_mana / n,
        'avg_price': total_price / n,
        'num_creatures': num_creatures,
        'num_spells': num_spells,
        'num_lands': num_lands,
    }
    for i in range(8):
        features[f'mana_bin_{i}'] = mana_bins[i]
    for ch in ['W','U','B','R','G']:
        features[f'color_{ch}'] = color_counts[ch]
    return features


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
        cards = r['cards_serialized'].split(';') if r.get('cards_serialized') else []
        all_cards.update(cards)
    vocab = [c for c, _ in all_cards.most_common(top_n_cards)]
    index = {c: i for i, c in enumerate(vocab)}

    X = []
    y = []
    for r in rows:
        vec = [0] * len(vocab)
        cards = r['cards_serialized'].split(';') if r.get('cards_serialized') else []
        for c in cards:
            if c in index:
                vec[index[c]] += 1
        # engineered features
        features = []
        # expected engineered columns: avg_mana, avg_price, num_creatures, num_spells, num_lands, mana_bin_0..7, color_W..color_G
        engineered_keys = ['avg_mana','avg_price','num_creatures','num_spells','num_lands'] + [f'mana_bin_{i}' for i in range(8)] + [f'color_{c}' for c in ['W','U','B','R','G']]
        for k in engineered_keys:
            features.append(float(r.get(k, 0)))
        X.append(vec + features)
        y.append(float(r.get('winrate', 0)))

    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X, y)
    # save vocab and engineered schema length so loader can reconstruct
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
    # no engineered features available at prediction time without card_db; require caller to pass them in separately
    try:
        return float(model.predict([vec])[0])
    except Exception:
        return 0.0


# ----------------- Simulation evaluation -----------------

def evaluate_candidate_sim(deck_cards: List[str], card_db: Dict[str, dict], n_games: int, commander: str, training_csv: str) -> float:
    deck_lines = collapse_deck_list(deck_cards, commander)
    deck_cards_sim = sim.parse_deck_list(deck_lines, card_db)
    decks = [deepcopy(deck_cards_sim) for _ in range(4)]
    results = sim.run_multiplayer_tournament(decks, n_games=n_games)
    wins = results.get('Player1', 0)
    winrate = wins / n_games
    # compute engineered features
    features = extract_deck_features(deck_cards, card_db)
    # append to training data with extra columns
    try:
        write_header = not os.path.exists(training_csv)
        with open(training_csv, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['commander','cards_serialized','winrate','avg_mana','avg_price','num_creatures','num_spells','num_lands'] + [f'mana_bin_{i}' for i in range(8)] + [f'color_{c}' for c in ['W','U','B','R','G']]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {'commander': commander, 'cards_serialized': ';'.join(deck_cards), 'winrate': winrate}
            row.update(features)
            writer.writerow(row)
    except Exception as e:
        print(f"[WARN] Could not write training data: {e}")
    return winrate


# ----------------- Active-learning genetic search -----------------

def run_search(cmdr_name: str, budget: float, n_games: int = 500, population_size: int = 6, generations: int = 5,
               candidate_pool_size: int = 200, fast_sims: int = 50, top_k_full: int = 5, retrain_every: int = 2,
               training_csv: str = 'training_data.csv', seed: int = None) -> dict:
    if seed is not None:
        random.seed(seed)
    print(f"[OPT-MLv2] Building baseline deck for {cmdr_name} with budget ${budget:.2f}...")
    baseline_lines, card_db = build_deck(cmdr_name, total_budget=budget, print_list=False)
    if not baseline_lines:
        print("[OPT-MLv2] Baseline build failed.")
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
        print('[OPT-MLv2] Loaded existing model; using it to pre-score candidates')

    best = None
    best_score = -1.0
    new_labels_since_retrain = 0

    for gen in range(1, generations + 1):
        print(f"[OPT-MLv2] Generation {gen}/{generations}")
        candidate_pool = []
        while len(candidate_pool) < candidate_pool_size:
            parent = random.choice(population)
            child = mutate_cards(parent, pool, card_db, budget, commander)
            if tuple(child) not in [tuple(x) for x in candidate_pool]:
                candidate_pool.append(child)

        model_scores = {}
        if model and vocab:
            for cand in candidate_pool:
                model_scores[tuple(cand)] = predict_deck_with_model(cand, model, vocab)
        else:
            for cand in candidate_pool:
                model_scores[tuple(cand)] = random.random()

        sorted_candidates = sorted(candidate_pool, key=lambda c: model_scores[tuple(c)], reverse=True)
        fast_selected = sorted_candidates[: min(len(sorted_candidates), population_size * 4)]

        fast_results = {}
        print(f"[OPT-MLv2] Running fast sims for {len(fast_selected)} candidates (sims={fast_sims})...")
        for cand in fast_selected:
            win = evaluate_candidate_sim(cand, card_db, n_games=fast_sims, commander=commander, training_csv=training_csv)
            fast_results[tuple(cand)] = win
            new_labels_since_retrain += 1

        candidates_for_full = sorted(fast_results.items(), key=lambda x: x[1], reverse=True)[:top_k_full]
        print(f"[OPT-MLv2] Running full sims for top {len(candidates_for_full)} candidates (sims={n_games})...")
        full_results = {}
        for tup, _ in candidates_for_full:
            cand = list(tup)
            win = evaluate_candidate_sim(cand, card_db, n_games=n_games, commander=commander, training_csv=training_csv)
            full_results[tuple(cand)] = win
            new_labels_since_retrain += 1
            if win > best_score:
                best_score = win
                best = deepcopy(cand)

        scored = []
        for cand in population:
            key = tuple(cand)
            score = full_results.get(key) or fast_results.get(key) or model_scores.get(key) or 0.0
            scored.append((score, cand))

        for key, val in list(full_results.items()) + list(fast_results.items()):
            scored.append((val, list(key)))

        scored.sort(key=lambda x: x[0], reverse=True)
        survivors = [ind for (_, ind) in scored[: max(1, len(scored)//2)]]

        new_pop = []
        while len(new_pop) < population_size:
            parent = random.choice(survivors)
            child = mutate_cards(parent, pool, card_db, budget, commander)
            if commander not in child:
                child[0] = commander
            new_pop.append(child)
        population = new_pop

        if SKLEARN_AVAILABLE and new_labels_since_retrain >= 20 and (gen % retrain_every == 0):
            print('[OPT-MLv2] Retraining model from', training_csv)
            model, vocab = retrain_model_from_csv(path=training_csv)
            new_labels_since_retrain = 0

    if best:
        best_lines = collapse_deck_list(best, commander)
        price = deck_price(best_lines, card_db)
        print('\n[OPT-MLv2] Best deck found:')
        for line in best_lines:
            print(line)
        print(f'Estimated price: ${price:.2f}')
        print(f'Estimated winrate (during search): {best_score:.3f}')
        return {"commander": cmdr_name, "deck": best_lines, "price": price, "winrate": best_score}
    else:
        print('[OPT-MLv2] No candidate found.')
        return {"commander": cmdr_name, "deck": None, "price": 0.0, "winrate": 0.0}
