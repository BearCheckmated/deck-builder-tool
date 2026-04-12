import requests
import time

# ---------------------------------------------------------
# 1. THE DATA FETCHERS
# ---------------------------------------------------------
def get_card_data(card_name):
    """Fetches a single card (used primarily for the Commander)."""
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
                "is_land": "land" in d.get('type_line', "").lower(),
                "cmc": d.get('cmc', 3.0)  
            }
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching {card_name}: {e}")
    return None

def get_bulk_card_data(card_names):
    """Fetches up to 75 cards at once using Scryfall's collection endpoint."""
    url = "https://api.scryfall.com/cards/collection"
    identifiers = [{"name": name} for name in card_names]
    
    try:
        # Scryfall requires a short delay between requests if making multiple
        time.sleep(0.1) 
        res = requests.post(url, json={"identifiers": identifiers})
        
        if res.status_code == 200:
            data = res.json()
            results = {}
            for d in data.get('data', []):
                results[d['name']] = {
                    "name": d.get('name'),
                    "price": float(d.get('prices', {}).get('usd') or 0),
                    "identity": d.get('color_identity', []),
                    "type": d.get('type_line', "").lower(),
                    "is_land": "land" in d.get('type_line', "").lower(),
                    # We need to add extras for the simulator...
                    "mana_cost":   d.get('mana_cost', ""),
                    "type_line":   d.get('type_line', ""),
                    "oracle_text": d.get('oracle_text', ""),
                    "power":       d.get('power', None),
                    "toughness":   d.get('toughness', None),
                }
            return results
        else:
            print(f"Failed to fetch batch. Status code: {res.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Network error during bulk fetch: {e}")
    
    return {}

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
    except requests.exceptions.RequestException: 
        return []

# ---------------------------------------------------------
# 2. THE LOGIC ENGINE
# ---------------------------------------------------------
def build_deck(cmdr_name=None, total_budget=None, print_list=True):
    """Builds a deck given the commander name and the total budget.
    
    Args:
        cmdr_name (str, optional): Name of the exact commander card. 
            Defaults to prompting user input.
        total_budget (float, optional): Maximum deck budget in USD.
            Defaults to prompting user input.
        print_list (boolean): Prints the list of cards.

    Returns:
        tuple: A tuple containing:
            - full_deck (list[str]): Deck list in Moxfield format,
              e.g. ["1 Lightning Bolt", "4 Forest"].
            - card_database (dict): Raw Scryfall data keyed by card name.

    Example:
        >>> deck_lines, card_db = build_deck(
        ...     cmdr_name="Cloud, Ex-SOLDIER",
        ...     total_budget=200
        ... )
    """
    if cmdr_name is None:
        cmdr_name = input("Commander: ")
    if total_budget is None:
        total_budget = float(input("Budget ($): "))
    
    # 1. SET TARGETS
    TARGET_TOTAL = 100
    
    
    c_info = get_card_data(cmdr_name)
    if not c_info:
        print("Commander not found!")
        return

    cmc = c_info.get('cmc', 3.0)
    TARGET_LANDS = round(30 + (cmc * 1.5))
    TARGET_LANDS = max(33, min(TARGET_LANDS, 40))  # clamp between 33–40
    # Use ASCII arrow to avoid Windows console encoding issues
    print(f"[AI] Commander CMC: {cmc:.0f} -> Using {TARGET_LANDS} lands")
    TARGET_SPELLS = TARGET_TOTAL - TARGET_LANDS - 1 

    identity = c_info['identity']
    rem_budget = total_budget - c_info['price']
    
    final_spells = []
    final_nonbasic_lands = []
    
    print(f"\n[AI] Fetching synergy data for {cmdr_name}...")
    raw_synergy = get_edhrec_synergy(cmdr_name)
    
    if not raw_synergy:
        print("No synergy data found. Check commander spelling.")
        return

    # 2. BULK FETCH SCRYFALL DATA
    # We grab the top 150 names (giving us a buffer for expensive/rejected cards)
    top_150_names = [item['name'] for item in raw_synergy[:150]]
    
    # Chunk into lists of 75 for Scryfall
    chunks = [top_150_names[i:i + 75] for i in range(0, len(top_150_names), 75)]
    
    print(f"[AI] Downloading card data in {len(chunks)} batches...")
    card_database = {}
    for chunk in chunks:
        batch_data = get_bulk_card_data(chunk)
        card_database.update(batch_data)

    # Ensure the commander is present in the local card database so downstream
    # simulators can look it up when parsing the deck list.
    if c_info and c_info.get('name'):
        card_database[c_info['name']] = {
            'name': c_info.get('name'),
            'price': c_info.get('price', 0),
            'identity': c_info.get('identity', []),
            'type_line': c_info.get('type', ''),
            'mana_cost': c_info.get('mana_cost', ''),
            'oracle_text': c_info.get('oracle_text', ''),
            'power': c_info.get('power', None),
            'toughness': c_info.get('toughness', None),
            'cmc': c_info.get('cmc', 0),
        }

    print("[AI] Assembling deck...\n")

    # 3. FILL SPELLS & UTILITY LANDS
    # We iterate over the original sorted synergy list to maintain priority
    for item in raw_synergy[:150]:
        if len(final_spells) >= TARGET_SPELLS and len(final_nonbasic_lands) >= 15:
            break
            
        # Pull from our local dictionary instead of the internet!
        card = card_database.get(item['name'])
        
        if not card or card['price'] == 0 or card['price'] > max(2, rem_budget / 3): 
            continue

        if card['is_land']:
            if len(final_nonbasic_lands) < 15 and item['name'] not in final_nonbasic_lands:
                final_nonbasic_lands.append(card['name'])
                rem_budget -= card['price']
                print(f"  + Utility Land: {card['name']} (${card['price']:.2f})")
        else:
            if len(final_spells) < TARGET_SPELLS and item['name'] not in final_spells:
                final_spells.append(card['name'])
                rem_budget -= card['price']
                print(f"  + Spell: {card['name']} (${card['price']:.2f})")

    # 4. CALCULATE BASICS
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

    # 5. FINAL ASSEMBLY
    # Use Scryfall's canonical name if available to avoid lookup mismatches
    full_deck = [f"1 {c_info.get('name', cmdr_name)} *CMDR*"]
    for s in final_spells: full_deck.append(f"1 {s}")
    for l in final_nonbasic_lands: full_deck.append(f"1 {l}")
    for b in basic_list: full_deck.append(b)

    # FINAL CHECK
    current_total = sum([int(line.split(' ', 1)[0]) for line in full_deck])
    if current_total < 100:
        shortfall = 100 - current_total
        full_deck[-1] = f"{int(full_deck[-1].split(' ')[0]) + shortfall} {full_deck[-1].split(' ', 1)[1]}"

    if print_list == True:
        print("\n" + "="*30 + "\nCLEAN MOXFIELD LIST\n" + "="*30)
        for line in full_deck:
            print(line)
        print(f"\nRemaining Budget: ${rem_budget:.2f}")

    # Returning the decklist to be used by other functions.
    return full_deck, card_database  

if __name__ == "__main__":
    build_deck()