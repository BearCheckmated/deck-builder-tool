Deployment instructions: systemd (Linux) and Windows

Systemd unit (recommended for Linux servers)
1. Copy deck-optimizer.service to /etc/systemd/system/
2. Edit the file: set User, WorkingDirectory, and ExecStart (use a virtualenv python if available)
3. Reload and enable:
   sudo systemctl daemon-reload
   sudo systemctl enable --now deck-optimizer.service
4. View logs:
   sudo journalctl -u deck-optimizer.service -f

Notes:
- Ensure Python deps are installed (pip install -r requirements.txt; for training also install scikit-learn and joblib).
- The optimizer writes training_data.csv into the working directory.

Windows (PowerShell background)
Start-Process -FilePath python -ArgumentList 'C:\path\to\deck-builder-tool\run_optimizer_batch.py --commander "Atraxa, praetor''s voice" --budget 100 --sims 2000 --pop 20 --gens 50' -NoNewWindow

Replace paths and parameters as needed.
