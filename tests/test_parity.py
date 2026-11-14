"""Clean-room numerical parity against committed ecotraj 1.2.2 observations."""
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE = json.loads((FIXTURES / "oracle_cases.json").read_text())
SIGNFLIP = json.loads((FIXTURES / "signflip_cases.json").read_text())


def _function(name, module="ecotrajpy"):
    # Resolve at test time so unfinished exports fail individually, never skip or
    # prevent collection of unrelated parity cases.
    try:
        fn = getattr(import_module(module), name)
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"Required API not yet implemented: {module}.{name}: {exc}", pytrace=False)
    if not callable(fn):
        pytest.fail(f"Required API not callable: {module}.{name}", pytrace=False)
    return fn


def _sequence(value):
    return value if isinstance(value, list) else [value]


# Circular sd = sqrt(-2 log rho). When the true value is 0 (a single angle), rho rounds to
# 1 or 1 - eps and the sd comes out as 0 or sqrt(2*eps) = 1.49e-08 depending on platform
# last-bit arithmetic. Only for sd columns, values at or below that floor count as equal.
SD_ULP_FLOOR = float(np.sqrt(2 * np.finfo(float).eps)) * 1.01


def _numeric(actual, expected, path):
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)  # JSON null (R NA/NaN) becomes NaN.
    assert a.shape == e.shape, f"{path}: shape {a.shape} != {e.shape}"
    if path.endswith(".sd"):
        floor = (np.abs(a) <= SD_ULP_FLOOR) & (np.abs(e) <= SD_ULP_FLOOR)
        a = np.where(floor, 0.0, a)
        e = np.where(floor, 0.0, e)
    np.testing.assert_allclose(a, e, rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=path)


def _labels(actual, expected, path):
    if expected is not None:
        assert list(map(str, actual)) == list(map(str, _sequence(expected))), path


def _compare(actual, expected, pool, path):
    if "ref" in expected:
        expected = pool[expected["ref"] - 1]
    kind = expected["kind"]
    if kind == "list":
        assert isinstance(actual, dict), f"{path}: expected dictionary"
        data = expected["data"]
        # Only a spelling change: the Python API calls R's p.value p_value.
        translated = {("p_value" if k == "p.value" else k): v for k, v in data.items()}
        assert set(actual) == set(translated), f"{path}: dictionary keys differ"
        for key, value in translated.items():
            _compare(actual[key], value, pool, f"{path}.{key}")
    elif kind == "frame":
        assert isinstance(actual, pd.DataFrame), f"{path}: expected DataFrame"
        # R automatic row names are 1-based; pandas defaults are 0-based.
        if not (isinstance(actual.index, pd.RangeIndex)
                and _sequence(expected["rows"]) == [str(i + 1) for i in range(len(actual))]):
            _labels(actual.index, expected["rows"], f"{path}: row labels")
        _labels(actual.columns, expected["columns"], f"{path}: column labels")
        for col in _sequence(expected["columns"]):
            values = _sequence(expected["data"][col])
            if any(isinstance(v, str) for v in values):
                assert actual[col].tolist() == values, f"{path}.{col}"
            else:
                _numeric(actual[col], values, f"{path}.{col}")
    elif kind == "matrix":
        # The convergence API permits arrays inside its result dictionary.
        if not path.startswith("trajectory_convergence/"):
            assert isinstance(actual, pd.DataFrame), f"{path}: expected labelled DataFrame"
        if isinstance(actual, pd.DataFrame):
            _labels(actual.index, expected["rows"], f"{path}: row labels")
            _labels(actual.columns, expected["columns"], f"{path}: column labels")
        _numeric(actual, expected["data"], path)
    else:
        assert kind == "vector", f"Unknown fixture kind: {kind}"
        if expected["names"] is not None and not path.startswith("trajectory_convergence/multiple"):
            assert isinstance(actual, pd.Series), f"{path}: expected named Series"
            _labels(actual.index, expected["names"], f"{path}: vector labels")
        _numeric(np.atleast_1d(actual), _sequence(expected["data"]), path)


# Documented, categorized deviations from ecotraj 1.2.2 (see tests/known_deviations.json
# and the parity table on the docs site). Marked xfail (non-strict) so they stay visible.
KNOWN = json.loads((Path(__file__).parent / "known_deviations.json").read_text())


def _param(case, output):
    pid = f"{case['id']}/{output['id']}"
    cat = KNOWN["ids"].get(pid)
    marks = [pytest.mark.xfail(reason=f"known deviation {cat}: {KNOWN['_categories'][cat]}", strict=False)] if cat else []
    return pytest.param(case, output, id=pid, marks=marks)


PARAMS = [_param(case, output) for case in ORACLE["cases"] for output in case["outputs"]]


@pytest.mark.parametrize("case,output", PARAMS)
def test_r_oracle(case, output):
    define = _function("define_trajectories")
    fn = _function(output["function_name"])
    inputs = case["input"]
    trajectory = define(np.asarray(inputs["d"], dtype=float), inputs["sites"],
                        surveys=inputs["surveys"], times=inputs["times"])
    kwargs = output["kwargs"] or {}
    try:
        if output["status"] == "error":
            # Require a deliberate input/operation error, not an unfinished stub
            # or incidental AttributeError/IndexError from a broken port.
            with pytest.raises((ValueError, TypeError, RuntimeError),
                               match=None) as caught:
                fn(trajectory, **kwargs)
            assert not isinstance(caught.value, NotImplementedError), (
                f"Required API not yet implemented: {output['function_name']}; "
                f"R error was: {output['error']}"
            )
        else:
            actual = fn(trajectory, **kwargs)
            _compare(actual, output["value"], case["pool"], output["id"])
    except NotImplementedError:
        pytest.fail(f"Required API not yet implemented: {output['function_name']}", pytrace=False)


@pytest.mark.parametrize("case", SIGNFLIP["cases"], ids=lambda case: case["id"])
def test_r_signflip(case):
    fn = _function("signflip_test", "ecotrajpy.nulls")
    try:
        result = fn(np.asarray(case["C"], dtype=float),
                    signs=np.asarray(case["signs"], dtype=int))
    except NotImplementedError:
        pytest.fail("Required API not yet implemented: ecotrajpy.nulls.signflip_test", pytrace=False)
    np.testing.assert_allclose(result["null"], case["null"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result["stat"], case["observed"], rtol=0, atol=1e-10)
    assert result["p_value"] == case["p"]
