"""List crops with strand_count <= 1 together with their packet's QC flags and overlay image.

Run from the repo root:  python scripts/check_low_counts.py
"""
import sys
from pathlib import Path

import pandas as pd

build = Path(sys.argv[1] if len(sys.argv) > 1 else "data_build")
max_count = int(sys.argv[2]) if len(sys.argv) > 2 else 1

d = pd.read_csv(build / "dataset.csv")
q = pd.read_csv(build / "qc_report.csv")

low = d[d["strand_count"] <= max_count][["packet", "position", "strand_count"]]
out = low.merge(q[["packet", "flags", "column_heights"]], on="packet", how="left")
out["flags"] = out["flags"].fillna("")
out["overlay"] = out["packet"].map(lambda p: f"{build}/contact/{p}_overlay.jpg")

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print(out.sort_values(["packet", "position"]).to_string(index=False))

flagged = out[out["flags"] != ""]["packet"].nunique()
print(f"\n{len(low)} crops with strand_count <= {max_count} in {out['packet'].nunique()} packets; "
      f"{flagged} of those packets have QC flags.")
