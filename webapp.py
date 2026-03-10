from flask import Flask, request
from deck_builder import build_deck
import os

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>MTG Commander Deck Builder</h1>
    <form action="/build">
        Commander Name:<br>
        <input name="commander"><br><br>

        Budget ($):<br>
        <input name="budget"><br><br>

        <button type="submit">Build Deck</button>
    </form>
    """

@app.route("/build")
def build():
    commander = request.args.get("commander")
    budget = float(request.args.get("budget"))

    deck, db = build_deck(cmdr_name=commander, total_budget=budget, print_list=False)

    return "<pre>" + "\n".join(deck) + "</pre>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)