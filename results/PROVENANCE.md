# Provenance of the Extension 2 results

- **Run 1** (2026-11-14, ~12:31–12:55 ET) at commit c297c65, pre-registered configuration
  (seed 2026111401, B = 199, R = 2000/1000). Later found to have used a stale numba on-disk cache
  for the ext2 null kernel (the cache did not notice that M2 had turned `directionality_coords_fast`
  into a Python wrapper; CI, which has no cache, failed on the same code).
- **Fix** at commit 53b8086: the kernel calls `ecotrajpy.fast._directionality_impl` directly and
  numba disk caching is off.
- **Run 2** (2026-11-14, ~13:00–13:20 ET) at 53b8086 with all caches cleared, same configuration:
  every per-replication CSV is **byte-identical** to run 1 (`cmp`), so the published results stand.
  Run 2's metadata is `ext2/run_meta_rerun_53b8086.json`.
- Independent spot-check: 3 replications recomputed with the slow reference statistic
  (`ecotrajpy.nulls.directionality_coords`) and hand-written P1 code gave identical statistics and counts.
