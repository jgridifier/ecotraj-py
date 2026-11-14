"""Run from repo root with PYTHONPATH=src and NUMBA_NUM_THREADS=8.

R and Python runs are sequential. Data interchange files live only in a
TemporaryDirectory. A fresh NUMBA_CACHE_DIR is created before importing Numba.
"""
import json
import os
from pathlib import Path
import platform
import subprocess
import shutil
import statistics
import tempfile
from time import perf_counter

_cache = tempfile.TemporaryDirectory(prefix="ecotraj-cold-cache-")
os.environ["NUMBA_CACHE_DIR"] = _cache.name

import numba
import numpy as np
import scipy

from ecotrajpy.fast import signflip_test_fast
from ecotrajpy.nulls import signflip_test
from panel import SEED, make_panel, make_signs

ROOT = Path(__file__).resolve().parent
R = os.environ.get('RSCRIPT') or shutil.which('Rscript') or '/home/box/mamba/envs/rscout/bin/Rscript'


def timed(fn, *args, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, perf_counter() - start


def repeated(fn, *args, **kwargs):
    samples = [timed(fn, *args, **kwargs) for _ in range(5)]
    return samples[-1][0], statistics.median(s[1] for s in samples), [s[1] for s in samples]


def main():
    numba.set_num_threads(8)
    tiny = make_panel(4, 3)
    tiny_signs = make_signs(4, 2)
    _, cold = timed(signflip_test_fast, tiny, signs=tiny_signs)
    _, warm, _ = repeated(signflip_test_fast, tiny, signs=tiny_signs)
    results = dict(seed=SEED, nsim=999, cores=8, numba_threads=numba.get_num_threads(),
                   python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                   numba=numba.__version__, platform=platform.platform(),
                   cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), ''),
                   affinity_cpus=len(os.sched_getaffinity(0)),
                   jit_first_call_seconds=cold, tiny_warm_call_seconds=warm,
                   jit_compile_overhead_estimate_seconds=cold-warm, cases=[])
    print(f'Cold JIT first call: {cold:.6f}s; warm tiny call: {warm:.6f}s', flush=True)
    for T, D in [(62, 14), (120, 50)]:
        C, signs = make_panel(T, D), make_signs(T)
        numba.set_num_threads(1)
        fast1, fast1_seconds, fast1_samples = repeated(signflip_test_fast, C, signs=signs)
        numba.set_num_threads(8)
        fast, fast_seconds, fast_samples = repeated(signflip_test_fast, C, signs=signs)
        np.testing.assert_array_equal(fast1['null'], fast['null'])
        reference, reference_seconds, reference_samples = repeated(signflip_test, C, signs=signs)
        np.testing.assert_allclose(fast['null'], reference['null'], rtol=0, atol=1e-12)
        assert fast['p_value'] == reference['p_value']
        print(f'T={T} D={D}: fast {fast_seconds:.3f}s, reference {reference_seconds:.3f}s', flush=True)
        with tempfile.TemporaryDirectory(prefix='ecotraj-bench-') as tmp:
            tmp = Path(tmp)
            np.savetxt(tmp/'C.csv', C, delimiter=',', fmt='%.17g')
            np.savetxt(tmp/'signs.csv', signs, delimiter=',', fmt='%.0f')
            subprocess.run([R, str(ROOT/'bench_r.R'), str(tmp/'C.csv'), str(tmp/'signs.csv'), str(tmp/'r.json')], check=True)
            r = json.loads((tmp/'r.json').read_text())
        r_null = np.array(r.pop('null'))
        r_error = float(np.max(np.abs(r_null-reference['null'][:len(r_null)])))
        np.testing.assert_allclose(r_null, reference['null'][:len(r_null)], rtol=0, atol=1e-12)
        np.testing.assert_allclose(r['observed'], reference['stat'], rtol=0, atol=1e-12)
        if not r['extrapolated']:
            assert r['p_value'] == reference['p_value']
        results['cases'].append(dict(T=T, D=D, reference_seconds=reference_seconds,
            fast_seconds=fast_seconds, fast1_seconds=fast1_seconds,
            fast_samples=fast_samples, fast1_samples=fast1_samples, reference_samples=reference_samples, stat=fast['stat'], p_value=fast['p_value'],
            max_abs_null_error=float(np.max(np.abs(fast['null']-reference['null']))),
            r_max_abs_null_error=r_error, r=r))
        (ROOT/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


if __name__ == '__main__':
    main()
