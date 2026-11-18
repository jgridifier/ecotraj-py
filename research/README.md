# Research material

This folder holds the **frozen pre-registered studies** that sit beside the
package: pre-registrations, simulation runners, and published results.
It is not part of the installable `ecotrajpy` package.

The scientific text of the pre-registrations is frozen. Path mentions inside
those files (`sims/`, `results/`, `prereg/`) are the layout at registration
time; they now resolve as follows:

| At registration | Current path |
|---|---|
| `prereg/` | `research/prereg/` |
| `sims/` | `research/sims/` |
| `results/` | `research/results/` |

Do not edit a pre-registration to change a criterion, seed, or design after
its run. Path-only updates in runners and in this README are layout, not a
new study.

## Layout

- `prereg/EXT2_PREREG.md` — Extension 2 (increment-level market/residual split). Merged before the run (PR #1, Addendum 1 PR #3).
- `prereg/P2_PREREG.md` — power curve for smaller swaps and fewer parts. Merged before the run (PR #11).
- `sims/` — data-generating process and runners. Defaults write under `research/results/`.
- `results/` — published outputs, summaries, and provenance. See also the [results page](https://jgridifier.github.io/ecotraj-py/results.html).

## Re-running (optional)

From the repository root, with the package installed (`pip install -e ".[test,fast]"`):

```bash
# Extension 2 (long; pre-registered configuration)
python research/sims/run_ext2.py
python research/sims/summarize.py

# P2 power curve (long)
python research/sims/run_p2.py
python research/sims/summarize_p2.py
```

Use `--reps` / `--seed` only for development; those flags mark the run as not
the pre-registered configuration.
