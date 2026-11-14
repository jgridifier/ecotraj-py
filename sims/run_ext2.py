"""Run the Extension 2 simulation (prereg/EXT2_PREREG.md).

Default = the pre-registered run (master seed 2026111401, R = 2000 / 1000, B = 199).
--reps / --seed override are for development only.
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")   # parallelise over replications instead

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgp import simulate  # noqa: E402
from ecotrajpy.ext2 import B_DRAWS, draw_signs, run_procedures  # noqa: E402

MASTER_SEED = 2026111401
CELLS = [("i", 40), ("i", 62), ("ii", 40), ("ii", 62), ("i-b", 40), ("i-b", 62),
         ("i-c", 40), ("i-c", 62), ("i-d", 40), ("i-d", 62)]
PRIMARY = {"i", "ii"}
PROCS = ["P1", "P1n", "L", "W", "Mk"]
COLUMNS = (["rep"] + [f"{p}_count" for p in PROCS] + [f"{p}_p" for p in PROCS]
           + [f"{p}_stat" for p in PROCS] + [f"{p}_valid" for p in PROCS]
           + ["realized_share", "mk_drift_to_noise", "M1_retained", "net_disp_u"])


def cell_name(scn, T):
    return f"{scn}_T{T}"


def one_rep(args):
    scn, T, entropy, spawn_key, r = args
    # identical to SeedSequence(master).spawn(10)[cell].spawn(R)[r]
    ss = np.random.SeedSequence(entropy, spawn_key=tuple(spawn_key) + (r,))
    rng = np.random.default_rng(ss)
    sim = simulate(scn, T, rng)
    signs = draw_signs(rng, B_DRAWS, T - 1)       # drawn after all DGP draws
    out = run_procedures(sim["S"], sim["X"], signs)
    info = out["info"]
    # retained-drift fraction and net displacement along u: scenario ii only (NaN elsewhere)
    ii = scn == "ii" and sim["u"] is not None
    net = float(info["E"].sum(axis=0) @ sim["u"]) if ii else np.nan
    retained = info["M1_retained"] if ii else np.nan
    nan = float("nan")
    row = [r] + [out[p]["count"] if out[p]["valid"] else nan for p in PROCS] \
        + [out[p]["p"] for p in PROCS] + [out[p]["stat"] for p in PROCS] \
        + [int(out[p]["valid"]) for p in PROCS] \
        + [sim["realized_share"], info["mk_drift_to_noise"], retained, net]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="all", help="comma list like i/40,ii/62 (default all)")
    ap.add_argument("--reps", type=int, default=None, help="DEV ONLY: override R")
    ap.add_argument("--seed", type=int, default=None, help="DEV ONLY: override master seed")
    ap.add_argument("--out", default="results/ext2")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    seed = MASTER_SEED if a.seed is None else a.seed
    children = np.random.SeedSequence(seed).spawn(len(CELLS))
    want = CELLS if a.cells == "all" else [(c.split("/")[0], int(c.split("/")[1]))
                                           for c in a.cells.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prereg = a.reps is None and a.seed is None and a.cells == "all"
    meta = dict(master_seed=seed, B=B_DRAWS, preregistered_configuration=prereg,
                reps_override=a.reps, cells={})
    t_all = time.time()
    with Pool(a.workers) as pool:
        for scn, T in want:
            idx = CELLS.index((scn, T))
            R = a.reps if a.reps is not None else (2000 if scn in PRIMARY else 1000)
            ss = children[idx]
            jobs = [(scn, T, ss.entropy, ss.spawn_key, r) for r in range(R)]
            t0 = time.time()
            rows = []
            for n, row in enumerate(pool.imap(one_rep, jobs, chunksize=4), 1):
                rows.append(row)
                if n % max(1, R // 10) == 0:
                    print(f"[{cell_name(scn, T)}] {n}/{R}  {time.time() - t0:.0f}s", flush=True)
            np.savetxt(out / f"{cell_name(scn, T)}.csv", np.array(rows), delimiter=",",
                       header=",".join(COLUMNS), comments="", fmt="%.10g")
            meta["cells"][cell_name(scn, T)] = dict(scenario=scn, T=T, R=R, cell_index=idx,
                                                    seconds=time.time() - t0)
    meta["total_seconds"] = time.time() - t_all
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("done", f"{meta['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()
