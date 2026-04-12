#!/usr/bin/env bash
set -euo pipefail

# setup_server.sh
# Usage:
#   bash setup_server.sh /absolute/path/to/repo username "Commander Name" 100 2000 20 50 /usr/bin/python3
# If arguments omitted, sensible defaults are used.

WORKDIR=${1:-$(pwd)}
USER=${2:-$(whoami)}
COMMANDER=${3:-"Atraxa, praetor's voice"}
BUDGET=${4:-100}
SIMS=${5:-2000}
POP=${6:-20}
GENS=${7:-50}
PYTHON=${8:-python3}

echo "[SETUP] Working directory: $WORKDIR"
echo "[SETUP] User: $USER"
echo "[SETUP] Commander: $COMMANDER"
echo "[SETUP] Budget: $BUDGET, sims=$SIMS, pop=$POP, gens=$GENS"

echo "\n[SETUP] Creating virtualenv..."
$PYTHON -m venv "$WORKDIR/venv"
# shellcheck source=/dev/null
source "$WORKDIR/venv/bin/activate"

echo "[SETUP] Upgrading pip and installing requirements..."
pip install --upgrade pip
if [ -f "$WORKDIR/requirements.txt" ]; then
    pip install -r "$WORKDIR/requirements.txt" || true
fi
# Install training deps
pip install scikit-learn joblib || true

SERVICE_PATH="$WORKDIR/deck-optimizer.service"
cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=Deck Optimizer
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${WORKDIR}
ExecStart=${WORKDIR}/venv/bin/python ${WORKDIR}/run_optimizer_batch.py --commander "${COMMANDER}" --budget ${BUDGET} --sims ${SIMS} --pop ${POP} --gens ${GENS}
Restart=on-failure
StandardOutput=append:${WORKDIR}/optimizer.log
StandardError=append:${WORKDIR}/optimizer.err

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "$SERVICE_PATH"

echo "\n[SETUP] Created systemd service file at: $SERVICE_PATH"

cat <<MSG
To install and start the service (requires sudo):

  sudo cp "$SERVICE_PATH" /etc/systemd/system/deck-optimizer.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now deck-optimizer.service
  sudo journalctl -u deck-optimizer.service -f

If you prefer to run manually instead of systemd, activate the venv and run:

  source "$WORKDIR/venv/bin/activate"
  python "$WORKDIR/run_optimizer_batch.py" --commander "${COMMANDER}" --budget ${BUDGET} --sims ${SIMS} --pop ${POP} --gens ${GENS}

MSG

echo "[SETUP] Done."
