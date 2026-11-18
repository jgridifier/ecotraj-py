"""Summarise Extension 2 results and evaluate the pre-registered criteria (section 6)."""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

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


EXPECTED_SEED = 2026111401
EXPECTED_B = 199
PRIMARY = {"i", "ii"}
DEV_BANNER = "DEVELOPMENT RUN - NOT A PREREGISTERED RESULT"


def expected_R(scn):
    return 2000 if scn in PRIMARY else 1000


def load(path):
    arr = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    return {n: arr[n] for n in arr.dtype.names}


def validate_meta(meta):
    """Provenance problems (empty list = preregistered configuration)."""
    if meta is None:
        return ["run_meta.json missing or unreadable"]
    bad = []
    if meta.get("preregistered_configuration") is not True:
        bad.append("preregistered_configuration is not true")
    if meta.get("master_seed") != EXPECTED_SEED:
        bad.append(f"master_seed is {meta.get('master_seed')}, expected {EXPECTED_SEED}")
    if meta.get("B") != EXPECTED_B:
        bad.append(f"B is {meta.get('B')}, expected {EXPECTED_B}")
    mc = meta.get("cells", {})
    for scn, T in ORDER:
        c = mc.get(f"{scn}_T{T}")
        if c is None:
            bad.append(f"run_meta.json has no entry for cell {scn}/{T}")
        elif c.get("R") != expected_R(scn):
            bad.append(f"run_meta R for {scn}/{T} is {c.get('R')}, expected {expected_R(scn)}")
    return bad


def validate_data(key, d):
    """CSV integrity problems for one cell."""
    bad, name = [], f"{key[0]}/{key[1]}"
    R = len(d["rep"])
    if R != expected_R(key[0]):
        bad.append(f"{name}: {R} rows, expected {expected_R(key[0])}")
    if len(np.unique(d["rep"])) != R:
        bad.append(f"{name}: replication ids are not unique")
    for p in PROCS:
        flag = d[f"{p}_valid"]
        if not np.all((flag == 0) | (flag == 1)):
            bad.append(f"{name}/{p}: valid flag is not 0/1")
            continue
        v = flag == 1
        c, pv = d[f"{p}_count"][v], d[f"{p}_p"][v]
        if not np.all(np.isfinite(c)) or np.any(c != np.round(c)) or np.any((c < 0) | (c > EXPECTED_B)):
            bad.append(f"{name}/{p}: counts are not integers in [0, {EXPECTED_B}]")
        elif not np.allclose(pv, (1 + c) / (EXPECTED_B + 1), rtol=0, atol=1e-9):
            bad.append(f"{name}/{p}: p-values are not (1 + count)/{EXPECTED_B + 1}")
        if np.any(np.isfinite(d[f"{p}_count"][~v])) or np.any(np.isfinite(d[f"{p}_p"][~v])):
            bad.append(f"{name}/{p}: invalid replications carry a count or p-value")
    return bad


def valid_mask(d, proc):
    return d[f"{proc}_valid"] == 1


def rate(d, proc):
    """Rejection rate over valid replications only; invalid ones are counted separately."""
    v = valid_mask(d, proc)
    rej = d[f"{proc}_count"][v] <= 9          # p <= 0.05 with B = 199
    x, n = int(rej.sum()), int(v.sum())
    if n == 0:
        return dict(rejections=0, R=0, R_total=len(v), invalid=len(v), rate=float("nan"),
                    ci95=[float("nan"), float("nan")])
    lo, hi = clopper_pearson(x, n)
    return dict(rejections=x, R=n, R_total=len(v), invalid=len(v) - n, rate=x / n, ci95=[lo, hi])


def in_band(r):
    return bool(BAND[0] <= r["rate"] <= BAND[1])       # False for NaN


def evaluate(cells, data):
    def r(scn, T, p):
        return cells[f"{scn}/{T}"]["rates"][p]

    crit = {}
    for p in ("P1", "L"):
        c1 = {T: in_band(r("i", T, p)) for T in (40, 62)}
        c2 = bool(r("ii", 62, p)["rate"] >= 0.80)
        crit[p] = dict(criterion1_size_i_both_T=bool(all(c1.values())),
                       size_by_T={str(T): bool(v) for T, v in c1.items()},
                       criterion2_power_ii_T62=c2,
                       passes_both=bool(all(c1.values()) and c2))
    crit["extension2_passes"] = crit["P1"]["passes_both"]       # criterion 3
    sec = {}
    for scn in ("i-b", "i-c", "i-d"):
        for T in (40, 62):
            rt = r(scn, T, "P1")
            ok = bool(rt["rate"] <= BAND[1]) if scn == "i-b" else in_band(rt)
            sec[f"{scn}/{T}"] = dict(rate=rt["rate"], ci95=rt["ci95"], meets=ok)
    crit["secondary_P1"] = sec
    d = data[("ii", 62)]                                             # criterion 7
    both = valid_mask(d, "P1") & valid_mask(d, "L")
    p1, l = d["P1_count"][both] <= 9, d["L_count"][both] <= 9
    b, c = int((p1 & ~l).sum()), int((~p1 & l).sum())
    pv = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    sizeok = crit["P1"]["criterion1_size_i_both_T"]
    crit["paired_ii_T62"] = dict(P1_rejects_L_not=b, L_rejects_P1_not=c, mcnemar_exact_p=pv,
                                 may_claim_P1_more_power_than_L=bool(sizeok and pv < 0.05 and b > c))
    return crit


def _fmt(r):
    return f"{r['rate']:.3f} (95% CI {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}; {r['rejections']}/{r['R']} valid)"


def verdict_lines(cells, crit):
    def r(scn, T, p):
        return cells[f"{scn}/{T}"]["rates"][p]

    L = ["## Verdict (pre-registered criteria, section 6)", ""]
    for p in ("P1", "L"):
        c = crit[p]
        L.append(f"**{p}**: {'MEETS' if c['passes_both'] else 'MISSES'} the criteria.")
        for T in (40, 62):
            ok = c["size_by_T"][str(T)]
            L.append(f"- Criterion 1 (size, scenario i, T={T}): {'meeting' if ok else 'missing'} "
                     f"the band [0.030, 0.070]; rate {_fmt(r('i', T, p))}.")
        ok = c["criterion2_power_ii_T62"]
        L.append(f"- Criterion 2 (power, scenario ii, T=62): {'meeting' if ok else 'missing'} "
                 f"the threshold >= 0.80; rate {_fmt(r('ii', 62, p))}.")
        L.append("")
    L.append(f"Extension 2 (criterion 3: P1 meets criteria 1 and 2): "
             f"**{'PASSES' if crit['extension2_passes'] else 'DOES NOT PASS'}**.")
    L.append("")
    if crit["L"]["passes_both"]:
        L += ["L meets both criteria, so the scout's diagnosis was not reproduced on this design.", ""]
    pr = crit["paired_ii_T62"]
    L.append(f"Paired comparison (descriptive, ii, T=62): P1 rejects and L does not in "
             f"{pr['P1_rejects_L_not']} replications; L rejects and P1 does not in "
             f"{pr['L_rejects_P1_not']}; exact McNemar p = {pr['mcnemar_exact_p']:.4g}. "
             + ("A statement that P1 has more power than L is permitted."
                if pr["may_claim_P1_more_power_than_L"]
                else "No statement that P1 has more power than L is permitted."))
    L.append("")
    miss = [k for k, v in crit["secondary_P1"].items() if not v["meets"]]
    L += ["## Secondary null misses (P1)", ""]
    if miss:
        L.append("**SECONDARY MISSES: " + ", ".join(miss) + "** (reported prominently; the primary "
                 "verdict above is unchanged).")
        L += [f"- {k}: rate {v['rate']:.3f} (95% CI {v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}) "
              f"misses {'<= 0.070' if k.startswith('i-b') else '[0.030, 0.070]'}"
              for k, v in crit["secondary_P1"].items() if not v["meets"]]
    else:
        L.append("No secondary null misses.")
    return L + [""]


def summarize(results, summary, md):
    rd = Path(results)
    try:
        meta = json.loads((rd / "run_meta.json").read_text())
    except (OSError, ValueError):
        meta = None
    prov = validate_meta(meta)
    integrity, cells, data = [], {}, {}
    for scn, T in ORDER:
        f = rd / f"{scn}_T{T}.csv"
        if not f.exists():
            integrity.append(f"{scn}/{T}: results CSV missing")
            continue
        d = data[(scn, T)] = load(f)
        integrity += validate_data((scn, T), d)
        c = dict(R=len(d["rep"]), rates={p: rate(d, p) for p in PROCS},
                 invalid={p: int((~valid_mask(d, p)).sum()) for p in PROCS},
                 realized_market_share_mean=float(np.mean(d["realized_share"])),
                 mk_drift_to_noise_mean=float(np.nanmean(d["mk_drift_to_noise"])))
        if scn == "ii":
            c["retained_drift_fraction_mean"] = float(np.nanmean(d["M1_retained"]))
            c["mean_residual_net_displacement_along_u"] = float(np.nanmean(d["net_disp_u"]))
        cells[f"{scn}/{T}"] = c

    problems = prov + integrity
    binding = not problems
    crit = evaluate(cells, data) if binding else None
    out = dict(binding=binding, development_run=bool(prov), provenance_problems=prov,
               integrity_problems=integrity, cells=cells,
               criteria=crit if binding else "withheld: not a valid preregistered result")
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).write_text(json.dumps(out, indent=2))

    L = []
    if prov:
        L += [f"# {DEV_BANNER}", ""]
    if problems:
        L += ["**No binding verdict is issued.** Problems:"] + [f"- {x}" for x in problems] + [""]
    L += ["| cell | R | " + " | ".join(PROCS) + " | mkt share | retained | net disp u |",
          "|---|---|" + "---|" * (len(PROCS) + 3)]
    for k, c in cells.items():
        rs = " | ".join(f"{c['rates'][p]['rate']:.3f} [{c['rates'][p]['ci95'][0]:.3f}, "
                        f"{c['rates'][p]['ci95'][1]:.3f}] (n={c['rates'][p]['R']})" for p in PROCS)
        ret = f"{c['retained_drift_fraction_mean']:.3f}" if "retained_drift_fraction_mean" in c else ""
        net = (f"{c['mean_residual_net_displacement_along_u']:.3f}"
               if "mean_residual_net_displacement_along_u" in c else "")
        L.append(f"| {k} | {c['R']} | {rs} | {c['realized_market_share_mean']:.3f} | {ret} | {net} |")
    L += ["", "## Invalid replications (non-finite statistic; excluded from rates, never counted as rejections)",
          "", "| cell | " + " | ".join(PROCS) + " |", "|---|" + "---|" * len(PROCS)]
    for k, c in cells.items():
        L.append(f"| {k} | " + " | ".join(str(c["invalid"][p]) for p in PROCS) + " |")
    L.append("")
    if binding:
        L += verdict_lines(cells, crit)
    Path(md).write_text("\n".join(L) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO_ROOT / "research" / "results" / "ext2"))
    ap.add_argument("--summary", default=str(REPO_ROOT / "research" / "results" / "ext2_summary.json"))
    ap.add_argument("--md", default=str(REPO_ROOT / "research" / "results" / "ext2_summary.md"))
    a = ap.parse_args(argv)
    summarize(a.results, a.summary, a.md)


if __name__ == "__main__":
    main()
