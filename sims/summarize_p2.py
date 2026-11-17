"""Summarize the P2 run (prereg/P2_PREREG.md) into results/P2_RESULTS.md and results/p2_summary.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

OUT = Path("results/p2")
meta = json.loads((OUT / "run_meta.json").read_text())
PROCS = ["P1", "P1n", "L", "W", "Mk"]


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


rows, summ = [], {}
for name, c in meta["cells"].items():
    d = pd.read_csv(OUT / f"{name}.csv")
    cell = dict(scenario=c["scenario"], D=c["D"], a=c["a"], T=c["T"], R=c["R"])
    for p in PROCS:
        v = d[f"{p}_valid"] == 1
        k = int((d.loc[v, f"{p}_count"] <= 9).sum()); n = int(v.sum())
        lo, hi = cp(k, n)
        cell[p] = dict(rate=k / n, lo=lo, hi=hi, n=n, invalid=int((~v).sum()))
    if c["scenario"] == "ii":
        cell["retained"] = float(d.M1_retained.mean()); cell["net_disp_u"] = float(d.net_disp_u.mean())
    cell["realized_share"] = float(d.realized_share.mean())
    summ[name] = cell
size = {T: summ[f"i_D6_a0_T{T}"]["P1"]["rate"] for T in (40, 62)}
size_ok = all(0.030 <= v <= 0.070 for v in size.values())
mdd = {}
for D in (14, 6):
    ok = [a for a in (0.025, 0.05, 0.075, 0.10) if summ[f"ii_D{D}_a{a:g}_T62"]["P1"]["rate"] >= 0.80]
    mdd[D] = min(ok) if ok else None
res = dict(cells=summ, size_check_D6=size, size_check_passes=size_ok, min_detectable_a_T62=mdd,
           master_seed=meta["master_seed"], B=meta["B"], total_seconds=meta["total_seconds"])
Path("results/p2_summary.json").write_text(json.dumps(res, indent=1))
L = ["# P2 results: Extension 2 power curve (pre-registered in prereg/P2_PREREG.md)", "",
     f"Master seed {meta['master_seed']}, B = {meta['B']}, R = 1,000 per cell, {meta['total_seconds']:.0f} s on 6 workers.", "",
     "| cell | P1 | P1n | L | W | Mk | retained | net disp u |", "|---|---|---|---|---|---|---|---|"]
for name, c in summ.items():
    f = lambda p: f"{c[p]['rate']:.3f} [{c[p]['lo']:.3f}, {c[p]['hi']:.3f}]"
    L.append(f"| {name} | {f('P1')} | {f('P1n')} | {f('L')} | {f('W')} | {f('Mk')} | {c.get('retained', float('nan')):.3f} | {c.get('net_disp_u', float('nan')):.3f} |")
L += ["", "## Verdict (pre-registered)", "",
      f"- Size check (P1, scenario i, D = 6): T = 40 {size[40]:.3f}, T = 62 {size[62]:.3f}; band [0.030, 0.070]: **{'PASS' if size_ok else 'MISS'}**.",
      f"- Minimum detectable swap at T = 62 (smallest a with P1 power >= 0.80): D = 14: {mdd[14] if mdd[14] is not None else 'none in grid'}; D = 6: {mdd[6] if mdd[6] is not None else 'none in grid'}.",
      "- Invalid replications are excluded from rates and never counted as rejections (counts in p2_summary.json).",
      "- No forecast metric."]
Path("results/P2_RESULTS.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
