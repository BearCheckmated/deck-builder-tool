import requests
import time

# ---------------------------------------------------------
# 1. THE DATA FETCHERS
# ---------------------------------------------------------
def get_card_data(card_name):
    """Enhanced Scryfall fetcher that identifies if a card is a Land."""
    formatted_name = card_name.replace(' ', '+')
    url = f"https://api.scryfall.com/cards/named?exact={formatted_name}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            d = res.json()
            return {
                "name": d.get('name'),
                "price": float(d.get('prices', {}).get('usd') or 0),
                "identity": d.get('color_identity', []),
                "type": d.get('type_line', "").lower(),
                "is_land": "land" in d.get('type_line', "").lower()
            }
    except: pass
    return None

def get_edhrec_synergy(commander):
    """Pulls cards sorted by synergy."""
    clean_name = commander.lower().replace(',', '').replace("'", "").replace(' ', '-')
    url = f"https://json.edhrec.com/pages/commanders/{clean_name}.json"
    try:
        data = requests.get(url).json()
        cards = []
        for clist in data.get('container', {}).get('json_dict', {}).get('cardlists', []):
            for c in clist.get('cardviews', []):
                if c.get('synergy'):
                    cards.append({'name': c['name'], 'synergy': c['synergy']})
        return sorted(cards, key=lambda x: x['synergy'], reverse=True)
    except: return []

# ---------------------------------------------------------
# 2. THE LOGIC ENGINE
# ---------------------------------------------------------
def build_deck():
    cmdr_name = input("Commander: ")
    total_budget = float(input("Budget ($): "))
    
    # 1. SET TARGETS
    TARGET_TOTAL = 100
    TARGET_LANDS = 37 # Standard EDH land count
    TARGET_SPELLS = TARGET_TOTAL - TARGET_LANDS - 1 # -1 for Commander
    
    c_info = get_card_data(cmdr_name)
    if not c_info:
        print("Commander not found!")
        return

    identity = c_info['identity']
    rem_budget = total_budget - c_info['price']
    
    final_spells = []
    final_nonbasic_lands = []
    
    print(f"\n[AI] Analyzing {cmdr_name}...")
    raw_synergy = get_edhrec_synergy(cmdr_name)
    
    # 2. FILL SPELLS & UTILITY LANDS
    for item in raw_synergy:
        # Stop if we are full
        if len(final_spells) >= TARGET_SPELLS and len(final_nonbasic_lands) >= 15:
            break
            
        card = get_card_data(item['name'])
        time.sleep(0.05) # Respect Scryfall
        if not card or card['price'] == 0 or card['price'] > (rem_budget / 5): 
            continue

        # Check if it's a land or a spell
        if card['is_land']:
            if len(final_nonbasic_lands) < 15 and item['name'] not in final_nonbasic_lands:
                final_nonbasic_lands.append(card['name'])
                rem_budget -= card['price']
                print(f"  + Utility Land: {card['name']} (${card['price']})")
        else:
            if len(final_spells) < TARGET_SPELLS and item['name'] not in final_spells:
                final_spells.append(card['name'])
                rem_budget -= card['price']
                print(f"  + Spell: {card['name']} (${card['price']})")

    # 3. CALCULATE BASICS

    basics_needed = TARGET_LANDS - len(final_nonbasic_lands)
    
    land_map = {'W': 'Plains', 'U': 'Island', 'B': 'Swamp', 'R': 'Mountain', 'G': 'Forest'}
    active_basics = [land_map[c] for c in identity if c in land_map] or ['Wastes']
    
    basic_list = []
    per_type = basics_needed // len(active_basics)
    extras = basics_needed % len(active_basics)

    for i, land_name in enumerate(active_basics):
        count = per_type + (1 if i < extras else 0)
        if count > 0:
            basic_list.append(f"{count} {land_name}")

    # 4. FINAL ASSEMBLY
    full_deck = [f"1 {cmdr_name} *CMDR*"]
    for s in final_spells: full_deck.append(f"1 {s}")
    for l in final_nonbasic_lands: full_deck.append(f"1 {l}")
    for b in basic_list: full_deck.append(b)

    # FINAL CHECK
    current_total = sum([int(line.split(' ', 1)[0]) for line in full_deck])
    if current_total < 100:
        shortfall = 100 - current_total
        full_deck[-1] = f"{int(full_deck[-1].split(' ')[0]) + shortfall} {full_deck[-1].split(' ', 1)[1]}"

    print("\n" + "="*30 + "\nCLEAN MOXFIELD LIST\n" + "="*30)
    for line in full_deck:
        print(line)

if __name__ == "__main__":
    build_deck()