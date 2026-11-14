import builtins
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ecotrajpy import fast
from ecotrajpy.nulls import directionality_coords, signflip_test


@pytest.mark.parametrize('T,p', [(3, 1), (4, 2), (10, 1), (20, 4), (35, 10), (62, 14), (70, 20)])
def test_random(T, p):
    rng = np.random.default_rng(1700 + T)
    C = rng.normal(size=(T, p)).cumsum(axis=0)
    signs = rng.choice([-1., 1.], size=(12, T - 1))
    actual = fast.signflip_test_fast(C, signs=signs)
    expected = signflip_test(C, signs=signs)
    np.testing.assert_allclose(actual['stat'], expected['stat'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual['null'], expected['null'], atol=1e-12, rtol=0)
    assert actual['p_value'] == expected['p_value']


CASES = json.loads((Path(__file__).parent / 'fixtures/signflip_cases.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_fixture(case):
    actual = fast.signflip_test_fast(case['C'], signs=case['signs'])
    expected = signflip_test(case['C'], signs=case['signs'])
    np.testing.assert_allclose(actual['null'], expected['null'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual['stat'], expected['stat'], atol=1e-12, rtol=0)
    assert actual['p_value'] == expected['p_value'] == case['p']


@pytest.mark.parametrize('C', [np.zeros((4, 2)), np.arange(6.)[:, None],
                            np.array([[0.], [1.], [0.], [1.]]), np.ones((2, 3))])
def test_degenerate(C):
    np.testing.assert_allclose(fast.directionality_coords_fast(C), directionality_coords(C), atol=1e-12, rtol=0)


def test_rng_and_ties():
    C = np.random.default_rng(7).normal(size=(5, 3))
    for signs in (None, np.array([[1.] * 4, [-1.] * 4])):
        a = fast.signflip_test_fast(C, nsim=20, rng=13, signs=signs)
        b = signflip_test(C, nsim=20, rng=13, signs=signs)
        np.testing.assert_allclose(a['null'], b['null'], atol=1e-12, rtol=0)
        assert a['p_value'] == b['p_value']
        assert a['p_value'] == (1 + np.sum(a['null'] >= a['stat'])) / (1 + len(a['null']))


def test_without_numba(monkeypatch):
    original = builtins.__import__
    def no_numba(name, *args, **kwargs):
        if name == 'numba':
            raise ImportError('simulated missing optional dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_numba)
    spec = importlib.util.spec_from_file_location('ecotrajpy._fallback_test', fast.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.njit is None
    for T, p in [(3, 1), (70, 20)]:
        C = np.random.default_rng(T).normal(size=(T, p)).cumsum(0)
        a = module.signflip_test_fast(C, nsim=4, rng=8)
        b = signflip_test(C, nsim=4, rng=8)
        np.testing.assert_allclose(a['null'], b['null'], atol=1e-12, rtol=0)
        assert a['p_value'] == b['p_value']


@pytest.mark.parametrize('T', range(3, 71))
def test_size_sweep(T):
    # Every survey count and every dimension from 1 through 20 is represented.
    for p in {1, 1 + (T - 3) % 20, 20}:
        C = np.random.default_rng(T * 100 + p).normal(size=(T, p)).cumsum(0)
        expected = directionality_coords(C)
        np.testing.assert_allclose(fast.directionality_coords_fast(C), expected,
                                   atol=1e-12, rtol=0)
        np.testing.assert_allclose(fast._directionality_numpy(C), expected,
                                   atol=1e-12, rtol=0)


def test_invalid_sign_shape():
    with pytest.raises(ValueError, match='signs'):
        fast.signflip_test_fast(np.zeros((4, 2)), signs=np.ones((2, 4)))


def test_empty_null():
    C = np.random.default_rng(90).normal(size=(5, 3))
    result = fast.signflip_test_fast(C, nsim=0)
    assert result['null'].shape == (0,)
    assert result['p_value'] == 1.0
