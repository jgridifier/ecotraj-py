"""Clean-room multiple-convergence observations from ecotraj 1.2.2."""
import json
from pathlib import Path

import numpy as np
import pytest

from ecotrajpy import define_trajectories, trajectory_convergence
from test_parity import _compare

ORACLE = json.loads(
    (Path(__file__).parent / "fixtures/convergence_roworder_cases.json").read_text()
)


def _define(case):
    data = case["input"]
    return define_trajectories(
        np.asarray(data["d"], dtype=float), data["sites"],
        surveys=data["surveys"], times=data["times"],
    )


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["id"])
def test_convergence_roworder_oracle(case):
    if case["status"] == "error":
        if case["stage"] == "define":
            with pytest.raises(ValueError, match="surveys.*numeric vector of integers"):
                _define(case)
        else:
            trajectory = _define(case)
            with pytest.raises(ValueError, match="trajectories must be synchronous"):
                trajectory_convergence(trajectory, type="multiple")
        return
    actual = trajectory_convergence(_define(case), type="multiple")
    # Reuse the parity comparator unchanged, including rtol=1e-8/atol=1e-10.
    _compare(actual, case["value"], [], "trajectory_convergence/multiple")
