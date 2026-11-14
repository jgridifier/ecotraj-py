"""Compare ecotrajpy against known ecotraj-computed values on private panels.

Usage: python scripts/parity_scout_panels.py --data-dir /path/to/data
(data is not part of this repo).
"""
import argparse
import os

import numpy as np
import pandas as pd

from ecotrajpy import clr, directionality_coords, increment_split

MARKET = {"A": ["vix", "nasdaq_ret4", "baa10y", "ta_ind_g1"],
          "E4": ["vix", "nasdaq_ret4", "baa10y", "eq_rev_ind_g4"]}
TARGETS = {
    "A": dict(dir=0.577301, resid=0.595092, market=0.377861, energy=0.161606),
    "E4": dict(dir=0.431413, resid=0.513292, market=0.329615, energy=None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    mq = pd.read_csv(os.path.join(a.data_dir, "market_q.csv"), index_col="observation_date")
    for panel in ("A", "E4"):
        df = pd.read_csv(os.path.join(a.data_dir, f"panel_{panel}.csv"), index_col=0)
        shares = df.div(df.sum(axis=1), axis=0)
        C = clr(shares.values)
        X = mq.loc[df.index, MARKET[panel]].values.astype(float)
        sp = increment_split(C, X)
        t = TARGETS[panel]
        rows = [("directionality", directionality_coords(C), t["dir"]),
                ("residual directionality", directionality_coords(sp["residual"]), t["resid"]),
                ("market directionality", directionality_coords(sp["market"]), t["market"])]
        if t["energy"] is not None:
            rows.append(("step_energy_market", sp["step_energy_market"], t["energy"]))
        print(f"panel {panel}")
        for name, ours, tgt in rows:
            print(f"  {name:26s} ours={ours:.6f} target={tgt:.6f} diff={ours-tgt:+.2e}")


if __name__ == "__main__":
    main()
