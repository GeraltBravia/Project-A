import sys
import pandas as pd
from pathlib import Path

# Danh sách các file cần gộp (tên file, đặt cùng thư mục với script)
BASE = Path(__file__).resolve().parent
files = [
	BASE / "XSS_dataset.csv",
	BASE / "Modified_SQL_Dataset.csv",
	BASE / "DDOS_dataset.csv",
]

dfs = []
for p in files:
	if not p.exists():
		print(f"WARNING: file not found: {p}")
		continue
	def try_read(path):
		# Try a list of encodings until one succeeds
		encodings = ["utf-8", "cp1252", "latin1"]
		for enc in encodings:
			try:
				try:
					return pd.read_csv(path, engine='python', encoding=enc, on_bad_lines='skip')
				except TypeError:
					return pd.read_csv(path, engine='python', encoding=enc)
			except Exception as e:
				print(f"read with encoding {enc} failed for {path.name}: {e}")
				continue
		return None

	df_part = try_read(p)
	if df_part is None:
		print(f"ERROR: could not read {p} with any tested encodings")
		continue
	print(f"Read {len(df_part)} rows from {p.name}")
	dfs.append(df_part)

if not dfs:
	print("No input files were read successfully. Exiting.")
	sys.exit(1)

# Gộp tất cả file
df = pd.concat(dfs, ignore_index=True)

out = BASE / "merged.csv"
df.to_csv(out, index=False)

print(f"Đã hợp nhất {len(dfs)} file CSV ({len(df)} tổng hàng) thành {out}")
