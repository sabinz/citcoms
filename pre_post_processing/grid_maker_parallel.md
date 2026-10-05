# Resumable oldest-first grid generation

The cascade implementation is contained in `grid_maker_parallel.py`; no new
runtime helper file is required. Run from the model directory with the existing cfg:

```bash
python /path/to/grid_maker_parallel.py grids.cfg
```

Keep the existing `procs = 2` setting at the top level of the cfg. It takes
precedence over the optional `workers` and `cores` aliases. The existing numeric
command-line argument still overrides the cfg. The limit applies to concurrent ages.
`multiprocess_depths` is ignored and reported: each age processes its requested
fields and depths sequentially to keep concurrency and memory use bounded.

## Folder names and the cascade

The existing final layout and filenames are preserved:

```text
model/temp/165/model_temp_t165_100.nc
model/temp_dimensional/165/model_temp_t165_100_dimensional.nc
model/temp_deviation/165/deviation_model_temp_t165_100.nc
```

While an age is incomplete, its field folders end in `.inprogress`:

```text
model/temp/165.inprogress/
model/temp_dimensional/165.inprogress/
model/temp_deviation/165.inprogress/
```

All requested field, depth, dimensional, deviation and enabled plot products
for the age must exist and have a strictly positive byte count before ANY of
its field folders are renamed to the final age. Checks use bytes; a small file
that a file browser rounds to “0 KB” is still nonempty. Manifests record the
configuration/source signature, expected filenames and sizes.

Requests in timesteps, Ma or Myr are resolved once to available source timesteps,
deduplicated, and ordered by actual reconstruction age from oldest to youngest.
All requested fields must be available at a selected timestep. Distinct
timesteps rounding to the same integer age, or levels/sections colliding on
output filenames, cause an error rather than an overwrite. The resolved
requests are printed before work starts.

At most the configured number of ages are running or waiting ahead of the oldest
unfinished age. A younger age can finish first, but remains `.inprogress` until
the older ages have been published. Temporary GMT/XYZ files and logs are isolated
in `model/gridmaker-work/<age>Ma.inprogress/`. Published-age logs are retained
in `<age>Ma.complete/` there.

The final layout has several field directories per age, so their renames are
individual atomic operations rather than a single atomic whole-age operation.
Every output is checked before the first rename. If interrupted between those
renames, matching manifests allow the remaining field renames to resume.

## Resuming

Run the same command again:

- Completed output is checked and skipped. New manifests must match current
  settings/source identities and recorded sizes.
- Validated temporary output is reused, including partial publication.
- Incomplete work and legacy final folders with missing/empty products are
  preserved under `.previous-<timestamp>-<id>` names, then rebuilt.
- Complete legacy folders without manifests are accepted using filename and
  positive-size checks only. This cannot establish that their settings match.
- Published manifests from different settings cause an error; those outputs
  are not overwritten. Use a separate output/model directory or explicitly move
  them aside when changing scientific settings.

On a worker failure, the command returns nonzero and stops scheduling new ages.
Running workers can finish into temporary folders for reuse. Ctrl-C waits for
active ages to finish into staging. Incomplete workspace locks protect work
left by a still-running child after its parent was forcibly terminated.

Output checks establish existence and nonzero size, not scientific correctness
or valid NetCDF content. Source identities use paths, sizes and modification
times. Keep the master outputs immutable during processing.

## Related audit fixes included

The cascade also fixes timestep requests being compared with ages and silently
skipped; unchecked worker exits; the undefined regional-model PID variable;
cwd-wide coordinate-cache reuse; shared temporary filenames; nested depth
pools; broad `KeyError` retry handling; and Boolean flags tested by presence.
Viscosity must be positive before applying `log10`.

Age workspaces isolate GMT state. Commands use `gmt` directly so GMT exit status
is preserved even on installations where `isogmt` hides it. Spherical
interpolation is attempted by default with a checked fallback to `surface`.
Sections specifying `Ll`, `Lu` or numeric `T` use `surface`, which implements
those options, rather than passing them to `sphinterpolate`.

## Checks and remaining priorities

```bash
python -W ignore::ResourceWarning -m unittest discover -s tests -v
```

Tests cover selection, bounded scheduling, zero-byte rejection and repair,
partial publication, resumption, source completeness, stale caches, collisions,
real spawned workers and GMT-generated analytic constant fields. One-worker and
two-worker grids are compared numerically. GMT tests are skipped when GMT is
unavailable. The warning filter suppresses existing Core-module file-handle
warnings.

Before a large scientific run, compare nonconstant representative fields with
known-correct grids, including grid geometry, finite/NaN masks, values and units.
Signed velocity components deserve a separate check: their mapping and scaling
include the framework's existing longitude/colatitude sign conventions. The
plate-frame helper still uses Python 2 syntax and needs a separate compatibility
update before enabling `make_plate_frame_grid` with Python 3.

Strict source row/header checks in the shared data reader and full-grid numerical
validation remain useful improvements. Memory and I/O scaling still need a
representative full-model benchmark; each age reads and assembles a full volume.
