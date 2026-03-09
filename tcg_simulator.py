"""
Really simple simulator of agents battling against each other to experiment upon deck_builder.py
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
# Using deck_builder...
from deck_builder import build_deck 

# Hard coded; doesn't appear to be fetched in Scryfall
BASIC_LANDS = {
    "Forest":   {"name": "Forest",   "mana_cost": "", "type_line": "Land", 
                 "oracle_text": "Add {G}.", "power": None, "toughness": None},
    "Mountain": {"name": "Mountain", "mana_cost": "", "type_line": "Land", 
                 "oracle_text": "Add {R}.", "power": None, "toughness": None},
    "Plains":   {"name": "Plains",   "mana_cost": "", "type_line": "Land", 
                 "oracle_text": "Add {W}.", "power": None, "toughness": None},
    "Island":   {"name": "Island",   "mana_cost": "", "type_line": "Land", 
                 "oracle_text": "Add {U}.", "power": None, "toughness": None},
    "Swamp":    {"name": "Swamp",    "mana_cost": "", "type_line": "Land", 
                 "oracle_text": "Add {B}.", "power": None, "toughness": None},
    "Wastes":   {"name": "Wastes",   "mana_cost": "", "type_line": "Land", 
                 "oracle_text": "Add {C}.", "power": None, "toughness": None},
}

# ---------------------------------------------------------------------------
# 1. CARD DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class Card:
    name: str
    mana_cost: int          # Simplified: just an integer mana value
    type_line: str          # "Land", "Creature", "Instant", "Sorcery"
    oracle_text: str = ""
    power: Optional[int] = None
    toughness: Optional[int] = None
    effect: Optional[str] = None   # "deal_damage", "draw_card", "heal"
    effect_value: int = 0          # How much damage/draw/heal

    @property
    def is_land(self):
        return "Land" in self.type_line

    @property
    def is_creature(self):
        return "Creature" in self.type_line

    @property
    def is_spell(self):
        return "Instant" in self.type_line or "Sorcery" in self.type_line


def parse_scryfall_card(data: dict) -> Card:
    """
    Convert a Scryfall-style JSON dict into a Card.
    Parses mana cost, type, and a small set of known oracle text patterns.
    """
    # --- Mana cost: count symbols like {R}, {1}, {G} etc.
    raw_cost = data.get("mana_cost", "")
    mana_cost = len([c for c in raw_cost.split("}") if c.strip("{").strip()])

    type_line = data.get("type_line", "")
    oracle_text = data.get("oracle_text", "")
    power = int(data["power"]) if data.get("power") not in (None, "*") else None
    toughness = int(data["toughness"]) if data.get("toughness") not in (None, "*") else None

    # --- Simple oracle text pattern matching for spells
    effect = None
    effect_value = 0

    text_lower = oracle_text.lower()
    if "deals" in text_lower and "damage" in text_lower:
        effect = "deal_damage"
        # Extract first number found in text
        for word in oracle_text.split():
            if word.isdigit():
                effect_value = int(word)
                break
    elif "draw" in text_lower and "card" in text_lower:
        effect = "draw_card"
        for word in oracle_text.split():
            if word.isdigit():
                effect_value = int(word)
                break
        if effect_value == 0:
            effect_value = 1
    elif "gain" in text_lower and "life" in text_lower:
        effect = "heal"
        for word in oracle_text.split():
            if word.isdigit():
                effect_value = int(word)
                break

    return Card(
        name=data["name"],
        mana_cost=mana_cost,
        type_line=type_line,
        oracle_text=oracle_text,
        power=power,
        toughness=toughness,
        effect=effect,
        effect_value=effect_value,
    )


# ---------------------------------------------------------------------------
# 2. CREATURE ON BOARD
# ---------------------------------------------------------------------------

@dataclass
class BoardCreature:
    """A creature card that has been played onto the board."""
    card: Card
    current_toughness: int = field(init=False)
    has_attacked: bool = False
    summoning_sick: bool = True  

    def __post_init__(self):
        self.current_toughness = self.card.toughness

    @property
    def is_alive(self):
        return self.current_toughness > 0

    @property
    def power(self):
        return self.card.power

    def take_damage(self, amount: int):
        self.current_toughness -= amount

    def __repr__(self):
        return (f"{self.card.name} "
                f"({self.card.power}/{self.current_toughness})"
                f"{'[sick]' if self.summoning_sick else ''}")


# ---------------------------------------------------------------------------
# 3. PLAYER STATE
# ---------------------------------------------------------------------------

class Player:
    def __init__(self, name: str, deck: list[Card]):
        self.name = name
        self.health = 20
        self.mana = 0
        self.max_mana = 0
        self.land_played_this_turn = False
        self.hand: list[Card] = []
        self.deck: list[Card] = list(deck)
        self.board: list[BoardCreature] = []
        random.shuffle(self.deck)
        self.draw(7)   # Opening hand, TODO: add mulligan if insufficient amount of lands

    def draw(self, n: int = 1):
        for _ in range(n):
            if self.deck:
                self.hand.append(self.deck.pop(0))

    def start_turn(self):
        self.mana = self.max_mana
        self.land_played_this_turn = False
        for creature in self.board:
            creature.has_attacked = False
            creature.summoning_sick = False
        self.draw(1)

    def play_land(self, card: Card):
        assert card.is_land and not self.land_played_this_turn
        self.hand.remove(card)
        self.max_mana += 1
        self.mana += 1
        self.land_played_this_turn = True

    def play_creature(self, card: Card):
        assert card.mana_cost <= self.mana
        self.hand.remove(card)
        self.mana -= card.mana_cost
        bc = BoardCreature(card=card)
        self.board.append(bc)

    def play_spell(self, card: Card, target_player: "Player",
                   target_creature: Optional[BoardCreature] = None):
        assert card.mana_cost <= self.mana
        self.hand.remove(card)
        self.mana -= card.mana_cost
        # Execute effect
        if card.effect == "deal_damage":
            if target_creature:
                target_creature.take_damage(card.effect_value)
                target_player.board = [c for c in target_player.board if c.is_alive]
            else:
                target_player.health -= card.effect_value
        elif card.effect == "draw_card":
            self.draw(card.effect_value)
        elif card.effect == "heal":
            self.health += card.effect_value

    def remove_dead_creatures(self):
        self.board = [c for c in self.board if c.is_alive]

    def __repr__(self):
        return (f"{self.name} | HP:{self.health} | Mana:{self.mana}/{self.max_mana} "
                f"| Hand:{len(self.hand)} | Board:{self.board}")


# ---------------------------------------------------------------------------
# 4. GAME STATE
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self, player1: Player, player2: Player):
        self.players = [player1, player2]
        self.turn = 0          # Which player's turn (0 or 1)
        self.round = 1
        self.game_over = False
        self.winner: Optional[str] = None

    @property
    def active_player(self) -> Player:
        return self.players[self.turn]

    @property
    def opponent(self) -> Player:
        return self.players[1 - self.turn]

    def check_win(self):
        for p in self.players:
            if p.health <= 0:
                self.game_over = True
                self.winner = self.players[1 - self.players.index(p)].name
        if not any(p.deck or p.hand for p in self.players):
            self.game_over = True
            self.winner = "Draw"

    def next_turn(self):
        self.turn = 1 - self.turn
        if self.turn == 0:
            self.round += 1
        self.active_player.start_turn()


# ---------------------------------------------------------------------------
# 5. COMBAT RESOLVER
# ---------------------------------------------------------------------------

def resolve_combat(attacker: Player, defender: Player, verbose: bool = False):
    """
    Simplified combat: each attacking creature deals damage to the
    defending player directly (no blocking for now).
    """
    attackers = [c for c in attacker.board
             if not c.has_attacked 
             and not c.summoning_sick
             and c.power is not None]
    if not attackers:
        return

    total_damage = sum(c.power for c in attackers)
    defender.health -= total_damage

    for c in attackers:
        c.has_attacked = True

    if verbose and attackers:
        names = ", ".join(str(c) for c in attackers)
        print(f"  ⚔  {attacker.name} attacks with [{names}] "
              f"→ {total_damage} damage to {defender.name} "
              f"(now {defender.health} HP)")


# ---------------------------------------------------------------------------
# 6. GREEDY AGENT
# ---------------------------------------------------------------------------

class GreedyAgent:
    """
    Heuristic-based agent. Priority order each turn:
      1. Play a land if possible (always good)
      2. Play the highest-power creature we can afford
      3. Cast damage spells at opponent's face
      4. Attack with all available creatures
      5. Cast draw/heal spells if mana remains
    """

    def __init__(self, player: Player):
        self.player = player

    def take_turn(self, state: GameState, verbose: bool = False):
        p = self.player
        opp = state.opponent

        if verbose:
            print(f"\n--- {p.name}'s turn (Round {state.round}) ---")
            print(f"  {p}")

        # 1. Play a land
        lands = [c for c in p.hand if c.is_land and not p.land_played_this_turn]
        if lands:
            p.play_land(lands[0])
            if verbose:
                print(f"Plays land: {lands[0].name} "
                      f"(mana now {p.max_mana})")

        # 2. Play creatures — highest power first, repeat while affordable
        while True:
            creatures = sorted(
                [c for c in p.hand if c.is_creature 
                and c.mana_cost <= p.mana
                and c.power is not None],  # TODO: no power is temporary, eventually we want them to declare attackers.
                key=lambda c: (c.power or 0),
                reverse=True
            )
            if not creatures:
                break
            chosen = creatures[0]
            p.play_creature(chosen)
            if verbose:
                print(f"Plays creature: {chosen.name} "
                      f"({chosen.power}/{chosen.toughness}) "
                      f"for {chosen.mana_cost} mana")

        # 3. Cast damage spells at opponent face
        while True:
            dmg_spells = sorted(
                [c for c in p.hand
                 if c.is_spell and c.effect == "deal_damage"
                 and c.mana_cost <= p.mana],
                key=lambda c: c.effect_value,
                reverse=True
            )
            if not dmg_spells:
                break
            spell = dmg_spells[0]
            p.play_spell(spell, opp)
            if verbose:
                print(f"Casts {spell.name}: "
                      f"{spell.effect_value} damage to {opp.name} "
                      f"(now {opp.health} HP)")

        # 4. Attack
        resolve_combat(p, opp, verbose=verbose)

        # 5. Draw spells with leftover mana
        while True:
            draw_spells = [c for c in p.hand
                           if c.is_spell and c.effect == "draw_card"
                           and c.mana_cost <= p.mana]
            if not draw_spells:
                break
            spell = draw_spells[0]
            p.play_spell(spell, opp)
            if verbose:
                print(f"Casts {spell.name}: draws {spell.effect_value}")


# ---------------------------------------------------------------------------
# 7. RANDOM AGENT (baseline for comparison)
# ---------------------------------------------------------------------------

class RandomAgent:
    """Plays legal actions in random order (baseline)."""

    def __init__(self, player: Player):
        self.player = player

    def take_turn(self, state: GameState, verbose: bool = False):
        p = self.player
        opp = state.opponent

        # Play a land randomly
        lands = [c for c in p.hand if c.is_land and not p.land_played_this_turn]
        if lands:
            p.play_land(random.choice(lands))

        # Play random affordable cards
        for _ in range(10):  # Safety limit
            playable = [c for c in p.hand
                        if not c.is_land and c.mana_cost <= p.mana]
            if not playable:
                break
            card = random.choice(playable)
            if card.is_creature:
                p.play_creature(card)
            elif card.is_spell:
                p.play_spell(card, opp)

        resolve_combat(p, opp, verbose=verbose)


# ---------------------------------------------------------------------------
# 8. GAME RUNNER
# ---------------------------------------------------------------------------

def run_game(deck1: list[Card], deck2: list[Card],
             agent1_cls=GreedyAgent, agent2_cls=GreedyAgent,
             verbose: bool = False,
             max_rounds: int = 30) -> str:
    """
    Run a single game between two agents. Returns winner's name.
    """
    p1 = Player("Player1", deck1)
    p2 = Player("Player2", deck2)
    state = GameState(p1, p2)

    agent1 = agent1_cls(p1)
    agent2 = agent2_cls(p2)
    agents = [agent1, agent2]

    p1.start_turn()

    while not state.game_over and state.round <= max_rounds:
        agents[state.turn].take_turn(state, verbose=verbose)
        state.check_win()
        if not state.game_over:
            state.next_turn()

    if not state.winner:
        # Determine winner by health at round limit
        if p1.health > p2.health:
            state.winner = p1.name
        elif p2.health > p1.health:
            state.winner = p2.name
        else:
            state.winner = "Draw"

    if verbose:
        print(f"\nWinner: {state.winner} "
              f"(P1: {p1.health} HP, P2: {p2.health} HP)")

    return state.winner


def run_tournament(deck1: list[Card], deck2: list[Card],
                   n_games: int = 100,
                   agent1_cls=GreedyAgent,
                   agent2_cls=GreedyAgent) -> dict:
    """
    Run N games and return win statistics.
    Useful for card evaluation — swap cards in/out and compare win rates.
    """
    results = {"Player1": 0, "Player2": 0, "Draw": 0}
    for _ in range(n_games):
        winner = run_game(deepcopy(deck1), deepcopy(deck2),
                          agent1_cls, agent2_cls, verbose=False)
        results[winner] = results.get(winner, 0) + 1

    total = n_games
    print(f"\n=== Tournament Results ({n_games} games) ===")
    for name, wins in results.items():
        print(f"  {name}: {wins} wins ({100*wins/total:.1f}%)")
    return results


# ---------------------------------------------------------------------------
# 9. DECK BUILDER COMPATIBILITY LAYER
# ---------------------------------------------------------------------------

def parse_deck_list(deck_lines: list[str],
                    card_database: dict) -> list[Card]:
    """
    Converts the output of your colleague's build_deck() into a list of
    Card objects usable by the simulator.
    """
    deck: list[Card] = []

    for line in deck_lines:
        line = line.strip()
        if not line:
            continue

        # --- Parse count and name ("4 Forest", "1 Atraxa *CMDR*")
        parts = line.split(" ", 1)
        try:
            count = int(parts[0])
        except ValueError:
            print(f"  [WARN] Could not parse line: {line!r}, skipping.")
            continue

        # Strip tags like *CMDR* from the name
        name = parts[1].replace("*CMDR*", "").strip()

        # --- Look up card data
        raw = card_database.get(name) or BASIC_LANDS.get(name)
        if not raw:
            print(f"  [WARN] '{name}' not found in card_database, skipping.")
            continue

        # --- Convert to simulator Card format
        # card_database entries use the same shape as Scryfall dicts,
        # so parse_scryfall_card() handles the heavy lifting.
        # We normalise a few field names your colleague may use differently.
        scryfall_dict = {
            "name":        raw.get("name", name),
            "mana_cost":   raw.get("mana_cost", ""),
            "type_line":   raw.get("type_line", ""),
            "oracle_text": raw.get("oracle_text", ""),
            "power":       raw.get("power", None),
            "toughness":   raw.get("toughness", None),
        }

        card = parse_scryfall_card(scryfall_dict)

        # Add `count` copies to the deck list
        for _ in range(count):
            deck.append(deepcopy(card))

    print(f"  [OK] Deck parsed: {len(deck)} cards total.")
    # Shuffles the cards
    random.shuffle(deck)
    return deck


def build_deck_and_simulate(deck_lines: list[str],
                             card_database: dict,
                             opponent_deck_lines: list[str] = None,
                             opponent_card_database: dict = None,
                             n_games: int = 100,
                             verbose_first_game: bool = True):
    """
    Convenience wrapper: takes build_deck() output for one or two players,
    runs a tournament, and returns win statistics.

    If no opponent deck is provided, mirrors deck1 as the opponent.
    """
    print("\n[SIM] Parsing Player 1 deck...")
    deck1 = parse_deck_list(deck_lines, card_database)

    if opponent_deck_lines and opponent_card_database:
        print("[SIM] Parsing Player 2 deck...")
        deck2 = parse_deck_list(opponent_deck_lines, opponent_card_database)
    else:
        print("[SIM] No opponent deck provided — mirroring Player 1.")
        deck2 = deepcopy(deck1)

    if verbose_first_game:
        print("\n[SIM] Running one verbose game first...\n")
        run_game(deepcopy(deck1), deepcopy(deck2),
                 agent1_cls=GreedyAgent, agent2_cls=GreedyAgent,
                 verbose=True)

    print(f"\n[SIM] Running {n_games}-game tournament...")
    return run_tournament(deck1, deck2, n_games=n_games)


# ---------------------------------------------------------------------------
# 10. EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    deck_lines, card_db = build_deck(
        cmdr_name="Cloud, Ex-SOLDIER", total_budget=200, print_list = False)

    opponent_deck_lines, opponent_card_database, = build_deck(
        cmdr_name="Zidane, Tantalus Thief", total_budget=200, print_list = False)

    build_deck_and_simulate(deck_lines, card_db, 
                            opponent_deck_lines,
                            opponent_card_database, 
                            n_games=500,
                            verbose_first_game=False)

