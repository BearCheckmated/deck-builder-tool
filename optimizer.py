"""
Simple Commander deck optimizer prototype.
- Prompts for commander name and USD budget (CLI)
- Builds a baseline deck using deck_builder.build_deck()
- Runs a small genetic search (mutations) evaluating candidates using tcg_simulator

Notes:
- This is a prototype that uses the existing 2-player simulator as a proxy for multiplayer EDH.
- Defaults are conservative (small population/generations). Increase on your server.
"""

from __future__ import annotations
import random
import time
import os
import csv
from copy import deepcopy
from typing import List, Dict

from deck_builder import build_deck, get_edhrec_synergy, get_bulk_card_data
import tcg_simulator as sim


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
    # Ensure commander is first and tagged
    out = []
    counts: Dict[str, int] = {}
    for c in cards:
        counts[c] = counts.get(c, 0) + 1
    # Commander should be exactly 1 copy
    if commander in counts:
        counts[commander] = 1
    out.append(f"1 {commander} *CMDR*")
    for name, cnt in counts.items():
        if name == commander:
            continue
        out.append(f"{cnt} {name}")
    return out


# ------------------ Optional model helpers ------------------
def load_trained_model(path: str = 'model.joblib'):
    try:
        import joblib
        m = joblib.load(path)
        print(f"[MODEL] Loaded model from {path}")
        return m.get('model'), m.get('vocab')
    except Exception:
        return None, None


def predict_deck_with_model(deck_cards: List[str], model, vocab: List[str]) -> float:
    if model is None or vocab is None:
        raise ValueError('Model or vocab not provided')
    index = {c: i for i, c in enumerate(vocab)}
    vec = [0] * len(vocab)
    for c in deck_cards:
        if c in index:
            vec[index[c]] += 1
    return float(model.predict([vec])[0])


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
    """Randomly swap 1-3 non-land, non-commander cards from the deck with others from the pool while keeping budget."""
    new_cards = cards[:]  # list of names
    # Identify indices that are mutable (not commander, not a basic land)
    mutable_idx = [i for i, c in enumerate(new_cards) if c != commander and c.lower() not in ("forest","island","swamp","mountain","plains","wastes")]
    if not mutable_idx:
        return new_cards
    n_swaps = random.randint(1, min(3, len(mutable_idx)))
    for _ in range(n_swaps):
        idx = random.choice(mutable_idx)
        # pick candidate from pool that's not already in deck and not commander
        candidates = [p for p in pool if p not in new_cards and p != commander]
        if not candidates:
            break
        choice = random.choice(candidates)
        old = new_cards[idx]
        new_cards[idx] = choice
        # Check budget; if exceeded, revert with some probability or try cheaper alternative
        tmp_lines = collapse_deck_list(new_cards, commander)
        price = deck_price(tmp_lines, card_db)
        if price > budget:
            # try to revert or replace with a cheaper option
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
        # create mutated variants of baseline
        candidate = mutate_cards(base_cards, pool, card_db, budget, commander)
        pop.append(candidate)
    return pop


def _simulate_multiplayer_game_once(decks_of_cards: list[list[sim.Card]], max_rounds: int = 50, verbose: bool = False) -> str:
    """Run a single multiplayer game using the simulator's Player and Card classes.
    Returns the winner's name (e.g., 'Player1') or 'Draw'.
    """
    # Create Player objects
    players = [sim.Player(f"Player{i+1}", deepcopy(deck)) for i, deck in enumerate(decks_of_cards)]

    # Start first player's turn
    players[0].start_turn()

    # Simple MultiGameState
    turn = 0
    round_num = 1
    game_over = False
    winner = None

    while not game_over and round_num <= max_rounds:
        p = players[turn]
        if p.health > 0:
            # Greedy play for this player (similar heuristics as simulator)
            # 1. Play a land
            lands = [c for c in p.hand if c.is_land and not p.land_played_this_turn]
            if lands:
                p.play_land(lands[0])
            # 2. Play creatures (highest power first)
            while True:
                creatures = sorted(
                    [c for c in p.hand if c.is_creature and c.mana_cost <= p.mana and c.power is not None],
                    key=lambda c: (c.power or 0),
                    reverse=True
                )
                if not creatures:
                    break
                chosen = creatures[0]
                p.play_creature(chosen)
            # 3. Choose a target opponent (highest HP)
            opponents = [opp for opp in players if opp is not p and opp.health > 0]
            target = max(opponents, key=lambda o: o.health) if opponents else None
            # 4. Cast damage spells at target
            while target:
                dmg_spells = sorted(
                    [c for c in p.hand if c.is_spell and c.effect == 'deal_damage' and c.mana_cost <= p.mana],
                    key=lambda c: c.effect_value,
                    reverse=True
                )
                if not dmg_spells:
                    break
                spell = dmg_spells[0]
                p.play_spell(spell, target)
            # 5. Attack target with creatures
            attackers = [c for c in p.board if not c.has_attacked and not c.summoning_sick and c.power is not None]
            if attackers and target:
                total_damage = sum(c.power for c in attackers)
                target.health -= total_damage
                for c in attackers:
                    c.has_attacked = True
            # 6. Draw spells
            while True:
                draw_spells = [c for c in p.hand if c.is_spell and c.effect == 'draw_card' and c.mana_cost <= p.mana]
                if not draw_spells:
                    break
                spell = draw_spells[0]
                p.play_spell(spell, target)

            # Clean up dead creatures
            for pl in players:
                pl.remove_dead_creatures()

        # Check terminal conditions
        alive = [pl for pl in players if pl.health > 0]
        if len(alive) == 1:
            game_over = True
            winner = alive[0].name
            break
        if not any(pl.deck or pl.hand for pl in players):
            game_over = True
            winner = 'Draw'
            break

        # Advance turn
        turn = (turn + 1) % len(players)
        if turn == 0:
            round_num += 1
        players[turn].start_turn()

    if not winner:
        alive = [pl for pl in players if pl.health > 0]
        if len(alive) == 1:
            winner = alive[0].name
        else:
            winner = 'Draw'
    return winner


def _run_multiplayer_tournament_local(decks_of_cards: list[list[sim.Card]], n_games: int = 100) -> dict:
    results = {f'Player{i+1}': 0 for i in range(len(decks_of_cards))}
    results['Draw'] = 0
    for _ in range(n_games):
        # deepcopy decks so each game starts fresh
        winner = _simulate_multiplayer_game_once([deepcopy(d) for d in decks_of_cards], max_rounds=50, verbose=False)
        results[winner] = results.get(winner, 0) + 1
    return results


def evaluate_candidate(deck_cards: List[str], card_db: Dict[str, dict], n_games: int, commander: str) -> float:
    """Evaluate a candidate using multiplayer simulations and append result to training CSV."""
    # Build deck lines and parse into simulator Card objects
    deck_lines = collapse_deck_list(deck_cards, commander)
    deck_cards_sim = sim.parse_deck_list(deck_lines, card_db)

    # Mirror candidate for 4-player games
    decks = [deepcopy(deck_cards_sim) for _ in range(4)]

    results = _run_multiplayer_tournament_local(decks, n_games=n_games)
    wins = results.get('Player1', 0)
    winrate = wins / n_games

    # Append training data
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



def run_search(cmdr_name: str, budget: float, n_games: int = 100, population_size: int = 6, generations: int = 5):
    print(f"[OPT] Building baseline deck for {cmdr_name} with budget ${budget:.2f}...")
    baseline_lines, card_db = build_deck(cmdr_name, total_budget=budget, print_list=False)
    if not baseline_lines:
        print("[OPT] Baseline build failed.")
        return {"commander": cmdr_name, "deck": None, "price": 0.0, "winrate": 0.0}

    # Build candidate pool from EDHREC synergy list + card_db keys
    raw = get_edhrec_synergy(cmdr_name)
    pool = [item['name'] for item in raw if item['name'] in card_db]
    # If pool is small, extend with card_db keys
    if len(pool) < 200:
        pool = list(card_db.keys())

    commander = baseline_lines[0].split(" ", 1)[1].replace("*CMDR*", "").strip()
    base_cards = expand_deck_lines(baseline_lines)
    # Ensure commander is first element for downstream code
    if commander in base_cards:
        # Move commander to index 0
        base_cards.remove(commander)
        base_cards.insert(0, commander)

    population = initial_population(baseline_lines, card_db, pool, commander, population_size, budget)

    # Try loading a trained model to help rank candidates (optional)
    model, vocab = load_trained_model()
    if model and vocab:
        print("[OPT] Model available — will print predicted scores for candidates.")

    best = None
    best_score = -1.0

    for gen in range(1, generations + 1):
        print(f"[OPT] Generation {gen}/{generations} — evaluating {len(population)} candidates...")
        scored = []

        # If model exists, show predictions to help prioritize evaluation
        if model and vocab:
            preds = [(predict_deck_with_model(ind, model, vocab), ind) for ind in population]
            preds.sort(key=lambda x: x[0], reverse=True)
            print("  [MODEL] Top predictions for this generation:")
            for pscore, pind in preds[: min(3, len(preds))]:
                print(f"    predicted={pscore:.3f} — sample card: {pind[0]}")

        for i, individual in enumerate(population):
            # Ensure commander is present at index 0
            if commander not in individual:
                individual[0] = commander
            score = evaluate_candidate(individual, card_db, n_games=n_games, commander=commander)
            scored.append((score, individual))
            print(f"  Candidate {i+1}: winrate={score:.3f}")
            # quick best tracking
            if score > best_score:
                best_score = score
                best = deepcopy(individual)

        # Select top half
        scored.sort(key=lambda x: x[0], reverse=True)
        survivors = [ind for (_, ind) in scored[: max(1, len(scored)//2)]]
        # Reproduce: fill population by mutating survivors
        new_pop = []
        while len(new_pop) < population_size:
            parent = random.choice(survivors)
            child = mutate_cards(parent, pool, card_db, budget, commander)
            # Ensure commander preserved
            if commander not in child:
                child[0] = commander
            new_pop.append(child)
        population = new_pop

    if best:
        best_lines = collapse_deck_list(best, commander)
        price = deck_price(best_lines, card_db)
        print("\n[OPT] Best deck found:")
        for line in best_lines:
            print(line)
        print(f"Estimated price: ${price:.2f}")
        print(f"Estimated winrate (during search): {best_score:.3f}")
        return {"commander": cmdr_name, "deck": best_lines, "price": price, "winrate": best_score}
    else:
        print("[OPT] No candidate found.")
        return {"commander": cmdr_name, "deck": None, "price": 0.0, "winrate": 0.0}


def main():
    print("Commander deck optimizer (prototype)")
    cmdr = input("Commander name: ").strip()
    if not cmdr:
        print("No commander given. Exiting.")
        return
    try:
        budget = float(input("Budget in USD (e.g. 500): ").strip())
    except Exception:
        print("Invalid budget. Exiting.")
        return
    try:
        n_games = int(input("Simulations per candidate (default 500): ").strip() or 500)
    except Exception:
        n_games = 500
    # Conservative defaults for quick feedback; increase on server
    pop = int(input("Population size (default 6): ").strip() or 6)
    gens = int(input("Generations (default 5): ").strip() or 5)

    start = time.time()
    run_search(cmdr, budget, n_games=n_games, population_size=pop, generations=gens)
    elapsed = time.time() - start
    print(f"\n[OPT] Search completed in {elapsed/60:.1f} minutes.")


if __name__ == '__main__':
    main()
