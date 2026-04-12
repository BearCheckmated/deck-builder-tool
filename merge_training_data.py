"""
Merge multiple per-run training CSVs into a single training_data.csv for model training.
Usage:
    python merge_training_data.py --out training_data.csv run1.csv run2.csv ...

This aligns columns and concatenates rows.
"""

import csv
import argparse
import os

parser = argparse.ArgumentParser(description='Merge training CSVs')
parser.add_argument('--out', default='training_data.csv')
parser.add_argument('inputs', nargs='+')
args = parser.parse_args()

all_fieldnames = []
rows = []
for p in args.inputs:
    if not os.path.exists(p):
        print('Missing:', p)
        continue
    with open(p, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not all_fieldnames:
            all_fieldnames = reader.fieldnames or []
        else:
            for fn in (reader.fieldnames or []):
                if fn not in all_fieldnames:
                    all_fieldnames.append(fn)
        for r in reader:
            rows.append(r)

with open(args.out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=all_fieldnames)
    writer.writeheader()
    for r in rows:
        # ensure all keys present
        out = {k: r.get(k, '') for k in all_fieldnames}
        writer.writerow(out)

print(f'Merged {len(rows)} rows into {args.out}')