"""Summarise Extension 2 results and evaluate the pre-registered criteria (section 6)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

PROCS = ["P1", "P1n", "L", "W", "Mk"]
BAND = (0.030, 0.070)
ORDER = [("i", 40), ("i", 62), ("ii", 40), ("ii", 62), ("i-b", 40), ("i-b", 62),
         ("i-c", 40), ("i-c", 62), ("i-d", 40), ("i-d", 62)]


def clopper_pearson(x, n, level=0.95):
    a = 1 - level
    lo = 0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def load(path):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {n: arr[n] for n in arr.dtype.names}


def rate(d, proc):
    rej = d[f"{proc}_count"] <= 9          # p <= 0.05 with B = 199
    x, n = int(rej.sum()), len(rej)
    lo, hi = clopper_pearson(x, n)
    return dict(rejections=x, R=n, rate=x / n, ci95=[lo, hi])


def in_band(r):
    return BAND[0] <= r["rate"] <= BAND[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ext2")
    ap.add_argument("--summary", default="results/ext2_summary.json")
    ap.add_argument("--md", default="results/ext2_summary.md")
    a = ap.parse_args()
    rd = Path(a.results)
    cells, data = {}, {}
    for scn, T in ORDER:
        f = rd / f"{scn}_T{T}.csv"
        if not f.exists():
            continue
        d = data[(scn, T)] = load(f)
        c = dict(R=len(d["rep"]), rates={p: rate(d, p) for p in PROCS},
                 realized_market_share_mean=float(np.mean(d["realized_share"])),
                 mk_drift_to_noise_mean=float(np.mean(d["mk_drift_to_noise"])))
        if scn == "ii":
            c["retained_drift_fraction_mean"] = float(np.mean(d["M1_retained"]))
            c["mean_residual_net_displacement_along_u"] = float(np.mean(d["net_disp_u"]))
        cells[f"{scn}/{T}"] = c

    def r(scn, T, p):
        return cells[f"{scn}/{T}"]["rates"][p]

    crit = {}
    for p in ("P1", "L"):
        try:
            c1 = {T: in_band(r("i", T, p)) for T in (40, 62)}
            c2 = r("ii", 62, p)["rate"] >= 0.80
        except KeyError:
            continue
        crit[p] = dict(criterion1_size_i_both_T=bool(all(c1.values())),
                       size_by_T={str(T): bool(v) for T, v in c1.items()},
                       criterion2_power_ii_T62=bool(c2),
                       passes_both=bool(all(c1.values()) and c2))
    if "P1" in crit:
        crit["extension2_passes"] = crit["P1"]["passes_both"]       # criterion 3
    sec = {}
    for scn in ("i-b", "i-c", "i-d"):
        for T in (40, 62):
            if f"{scn}/{T}" in cells:
                rt = r(scn, T, "P1")
                ok = rt["rate"] <= BAND[1] if scn == "i-b" else in_band(rt)
                sec[f"{scn}/{T}"] = dict(rate=rt["rate"], ci95=rt["ci95"], meets=bool(ok))
    crit["secondary_P1"] = sec
    if ("ii", 62) in data:                                           # criterion 7
        d = data[("ii", 62)]
        p1, l = d["P1_count"] <= 9, d["L_count"] <= 9
        b, c = int((p1 & ~l).sum()), int((~p1 & l).sum())
        pv = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
        sizeok = crit.get("P1", {}).get("criterion1_size_i_both_T", False)
        crit["paired_ii_T62"] = dict(P1_rejects_L_not=b, L_rejects_P1_not=c, mcnemar_exact_p=pv,
                                     may_claim_P1_more_power_than_L=bool(sizeok and pv < 0.05
                                                                         and b > c))
    out = dict(cells=cells, criteria=crit)
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(out, indent=2))

    L = ["| cell | R | " + " | ".join(PROCS) + " | mkt share | retained | net disp u |",
         "|---|---|" + "---|" * (len(PROCS) + 3)]
    for k, c in cells.items():
        rs = " | ".join(f"{c['rates'][p]['rate']:.3f} [{c['rates'][p]['ci95'][0]:.3f}, "
                        f"{c['rates'][p]['ci95'][1]:.3f}]" for p in PROCS)
        ret = f"{c['retained_drift_fraction_mean']:.3f}" if "retained_drift_fraction_mean" in c else ""
        net = (f"{c['mean_residual_net_displacement_along_u']:.3f}"
               if "mean_residual_net_displacement_along_u" in c else "")
        L.append(f"| {k} | {c['R']} | {rs} | {c['realized_market_share_mean']:.3f} | {ret} | {net} |")
    L += ["", "## Criteria", "```json", json.dumps(crit, indent=2), "```"]
    Path(a.md).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
