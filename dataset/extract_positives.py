import pandas as pd
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
merged_path = BASE / "merged.csv"
# If merged.csv doesn't exist, try to build it from the known files
if not merged_path.exists():
    files = [BASE / "XSS_dataset.csv", BASE / "Modified_SQL_Dataset.csv", BASE / "DDOS_dataset.csv"]
    dfs = []
    for p in files:
        if not p.exists():
            print(f"WARNING: input not found: {p}")
            continue
        # try common encodings
        for enc in ("utf-8", "cp1252", "latin1"):
            try:
                dfp = pd.read_csv(p, engine='python', encoding=enc, on_bad_lines='skip')
                print(f"Read {len(dfp)} rows from {p.name} using {enc}")
                dfs.append(dfp)
                break
            except TypeError:
                dfp = pd.read_csv(p, engine='python', encoding=enc)
                print(f"Read {len(dfp)} rows from {p.name} using {enc} (fallback)")
                dfs.append(dfp)
                break
            except Exception as e:
                print(f"Failed reading {p.name} with {enc}: {e}")
                continue
    if not dfs:
        print("No input files readable. Exiting.")
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True)
else:
    df = pd.read_csv(merged_path, engine='python')

# Ensure Label column exists (case-insensitive)
label_col = None
for c in df.columns:
    if str(c).lower() == 'label':
        label_col = c
        break
if label_col is None:
    print('No Label column found in input. Available columns:', df.columns.tolist())
    sys.exit(1)

# Filter rows where Label == 1 (numeric or string)
positives = df[df[label_col].astype(str).str.strip() == '1']
out = BASE / 'merged_positive.csv'
positives.to_csv(out, index=False)
print(f"Wrote {len(positives)} positive rows to {out}")
