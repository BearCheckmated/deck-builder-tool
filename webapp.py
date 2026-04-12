import os
import sys
import signal
import subprocess
from flask import Flask, request, redirect, url_for, send_file, abort
from deck_builder import build_deck

app = Flask(__name__)

ROOT = os.path.dirname(__file__)
LOG = os.path.join(ROOT, 'optimizer.log')
ERR = os.path.join(ROOT, 'optimizer.err')
RESULTS = os.path.join(ROOT, 'results')
TRAINING = os.path.join(ROOT, 'training_data.csv')
MODEL = os.path.join(ROOT, 'model.joblib')
PIDFILE = os.path.join(ROOT, 'run.pid')


def get_training_rows():
    if not os.path.exists(TRAINING):
        return 0
    try:
        with open(TRAINING, 'r', encoding='utf-8') as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def model_exists():
    return os.path.exists(MODEL)


def list_decks():
    if not os.path.isdir(RESULTS):
        return []
    out = []
    for fn in sorted(os.listdir(RESULTS), key=lambda f: os.path.getmtime(os.path.join(RESULTS, f)), reverse=True):
        if fn.endswith('.deck'):
            out.append(fn)
    return out


def tail(path, n_chars=20000):
    if not os.path.exists(path):
        return ''
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            data = f.read()
            return data[-n_chars:]
    except Exception:
        return ''


def start_run(commander, budget, sims, pop, gens, training_csv, seed):
    if os.path.exists(PIDFILE):
        return False, 'Process already running'
    cmd = [sys.executable, os.path.join(ROOT, 'run_optimizer_batch.py'), '--commander', commander, '--budget', str(budget), '--sims', str(sims), '--pop', str(pop), '--gens', str(gens), '--training-csv', training_csv]
    if seed is not None:
        cmd += ['--seed', str(seed)]
    out = open(LOG, 'a')
    err = open(ERR, 'a')
    # start_new_session works on Unix; on Windows it is ignored but available in Python 3.8+
    proc = subprocess.Popen(cmd, stdout=out, stderr=err, cwd=ROOT, start_new_session=True)
    with open(PIDFILE, 'w') as f:
        f.write(str(proc.pid))
    return True, f'Started pid {proc.pid}'


def stop_run():
    if not os.path.exists(PIDFILE):
        return False, 'No running process'
    try:
        with open(PIDFILE, 'r') as f:
            pid = int(f.read().strip())
        # politely terminate
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        os.remove(PIDFILE)
        return True, f'Stopped pid {pid}'
    except Exception as e:
        return False, str(e)


@app.route('/')
def index():
    rows = get_training_rows()
    has_model = model_exists()
    decks = list_decks()
    running = os.path.exists(PIDFILE)
    html = f"""
    <h1>Deck Builder Dashboard</h1>
    <p>Training rows: {rows}</p>
    <p>Model exists: {has_model}</p>
    <p>Optimizer running: {running}</p>
    <h2>Start a run</h2>
    <form action="/start" method="post">
      Commander: <input name="commander" value="Atraxa, Praetors' Voice"><br>
      Budget USD: <input name="budget" value="100"><br>
      Sims per candidate (full): <input name="sims" value="500"><br>
      Population: <input name="pop" value="20"><br>
      Generations: <input name="gens" value="50"><br>
      Training CSV: <input name="training_csv" value="training_run.csv"><br>
      Seed (optional): <input name="seed" value=""><br>
      <input type="submit" value="Start">
    </form>
    <form action="/stop" method="post"><button>Stop run</button></form>

    <h2>Logs (tail)</h2>
    <pre style="height:300px;overflow:auto;border:1px solid #ccc;">{tail(LOG)}</pre>

    <h2>Recent decks</h2>
    <ul>
    """
    for d in decks[:50]:
        html += f'<li><a href="/deck/{d}">{d}</a></li>'
    html += '</ul>'
    html += "<h2>Build quick deck</h2>\n<form action=\"/build\" method=\"get\">Commander: <input name=\"commander\"> Budget: <input name=\"budget\"> <button>Build</button></form>"
    return html


@app.route('/start', methods=['POST'])
def route_start():
    commander = request.form.get('commander')
    budget = float(request.form.get('budget') or 0)
    sims = int(request.form.get('sims') or 500)
    pop = int(request.form.get('pop') or 6)
    gens = int(request.form.get('gens') or 5)
    training_csv = request.form.get('training_csv') or 'training_run.csv'
    seed = request.form.get('seed')
    seed_val = int(seed) if seed else None
    ok, msg = start_run(commander, budget, sims, pop, gens, training_csv, seed_val)
    return redirect(url_for('index'))


@app.route('/stop', methods=['POST'])
def route_stop():
    ok, msg = stop_run()
    return redirect(url_for('index'))


@app.route('/deck/<path:name>')
def view_deck(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=False)


@app.route('/training')
def download_training():
    if not os.path.exists(TRAINING):
        abort(404)
    return send_file(TRAINING, as_attachment=True)


@app.route('/build')
def build_route():
    commander = request.args.get('commander')
    try:
        budget = float(request.args.get('budget') or 100)
    except Exception:
        budget = 100
    if not commander:
        return redirect(url_for('index'))
    deck, db = build_deck(cmdr_name=commander, total_budget=budget, print_list=False)
    return '<pre>' + '\n'.join(deck) + '</pre>'


if __name__ == '__main__':
    os.makedirs(RESULTS, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
