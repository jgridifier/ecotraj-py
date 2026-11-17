"""Run the P2 power curve (prereg/P2_PREREG.md). Same DGP and procedures as Extension 2; only a and D vary.

Default = the pre-registered run (master seed 2026111701, R = 1000, B = 199, 18 cells in the stated order).
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dgp  # noqa: E402
from ecotrajpy.ext2 import B_DRAWS, draw_signs, run_procedures  # noqa: E402
from run_ext2 import COLUMNS, PROCS  # noqa: E402

MASTER_SEED = 2026111701
CELLS = [("i", 6, 0.0, 40), ("i", 6, 0.0, 62)] + [
    ("ii", D, a, T) for D in (14, 6) for a in (0.025, 0.05, 0.075, 0.10) for T in (40, 62)]


def set_dgp(D, a):
    dgp.D = D
    dgp.SIGMA = 0.4 / np.sqrt(D - 1)
    dgp.TAU2 = 0.35 * dgp.SIGMA**2 / (0.65 * dgp.K)
    dgp.A_DRIFT = a


def name(c):
    scn, D, a, T = c
    return f"{scn}_D{D}_a{a:g}_T{T}"


def one_rep(args):
    scn, D, a, T, entropy, spawn_key, r = args
    set_dgp(D, a if a > 0 else 0.10)
    ss = np.random.SeedSequence(entropy, spawn_key=tuple(spawn_key) + (r,))
    rng = np.random.default_rng(ss)
    sim = dgp.simulate(scn, T, rng)
    signs = draw_signs(rng, B_DRAWS, T - 1)
    out = run_procedures(sim["S"], sim["X"], signs)
    info = out["info"]
    ii = scn == "ii" and sim["u"] is not None
    net = float(info["E"].sum(axis=0) @ sim["u"]) if ii else np.nan
    retained = info["M1_retained"] if ii else np.nan
    nan = float("nan")
    return ([r] + [out[p]["count"] if out[p]["valid"] else nan for p in PROCS]
            + [out[p]["p"] for p in PROCS] + [out[p]["stat"] for p in PROCS]
            + [int(out[p]["valid"]) for p in PROCS]
            + [sim["realized_share"], info["mk_drift_to_noise"], retained, net])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None, help="DEV ONLY")
    ap.add_argument("--out", default="results/p2")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    R = a.reps or 1000
    children = np.random.SeedSequence(MASTER_SEED).spawn(len(CELLS))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = dict(master_seed=MASTER_SEED, B=B_DRAWS, R=R, preregistered_configuration=a.reps is None, cells={})
    t_all = time.time()
    with Pool(a.workers) as pool:
        for idx, c in enumerate(CELLS):
            ss = children[idx]
            jobs = [(*c, ss.entropy, ss.spawn_key, r) for r in range(R)]
            t0 = time.time()
            rows = list(pool.imap(one_rep, jobs, chunksize=4))
            np.savetxt(out / f"{name(c)}.csv", np.array(rows), delimiter=",", header=",".join(COLUMNS),
                       comments="", fmt="%.10g")
            meta["cells"][name(c)] = dict(scenario=c[0], D=c[1], a=c[2], T=c[3], R=R, cell_index=idx,
                                         seconds=time.time() - t0)
            print(f"[{name(c)}] done {time.time() - t0:.0f}s", flush=True)
    meta["total_seconds"] = time.time() - t_all
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("done", f"{meta['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()
