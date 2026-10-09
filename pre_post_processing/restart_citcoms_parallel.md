# Parallel dynamic-topography restart preparation

Run from the master model directory, as with `restart_citcoms.py`:

```bash
python /path/to/restart_citcoms_parallel.py restart.cfg
```

Use [the example configuration](sample_data/restart_citcoms_parallel.cfg), based
on the supplied 165–0 Ma configuration. Set `workers = 2` (or another positive
integer) in that file. This is the maximum number of concurrently prepared ages,
not the number of MPI ranks for the subsequent CitcomS solver runs. The script
prepares inputs only; it does not launch CitcomS. It supports ASCII processor
velocity files, dynamic topography, and the `separate` folder structure.

`master_run_cfg` provides the solver input template. `master_run_pid` provides
the source data location, time table, mesh, and restart `coor_file` reference.
An explicit `CitcomS.*.coor_file` override takes precedence over that reference.
Paths in both configurations are
relative to the master model working directory. Framework defaults and the
master coordinate/time files must be available as for the serial workflow.

## Scheduling and completed folders

Requested ages select only available velocity-file timesteps whose actual ages
round to the requested age's integer Ma label. For example, 389.962 Ma matches
390 Ma, but is never used for a missing 385 Ma output. Missing ages are logged
and skipped; if none match, the script reports an error. Among matching outputs,
the closest age is selected; ties choose the younger age, then the lowest step.
A request for 400 Ma therefore prefers an available exact 400 Ma initial state
over step 1 at 399.937 Ma.

Selections are reported and sorted oldest first. Requests selecting the same
step are merged. Distinct selected steps sharing a rounded age (possible with
fractional-age requests) receive `_step<TIMESTEP>` folder suffixes. Noncolliding
folder names and the existing rounded solver start-age convention are unchanged.
All processor inputs for a selected timestep must exist and be nonempty; partial
source output fails preflight rather than generating an incomplete restart.

Each age is built inside `restart_dynamic_topography_<age>Ma.inprogress/`:

- `ic_dir/` contains the modified processor velocity/temperature files.
- `Age<age>Ma/` is the directory for the future solver outputs.
- `<model>_dynamic_topography_<age>Ma.input` is the solver input.
- `prepare.log` records the age's processing, including errors.
- `restart_complete.json` records the completed validation and output checksums.

The parent publishes completed ages strictly oldest to youngest, by renaming
the staging directory to `restart_dynamic_topography_<age>Ma`. At most `workers`
ages are active or waiting ahead of the oldest unfinished age, so a slow age
cannot leave an unbounded amount of younger work waiting on disk.

The temperature transformation uses the existing cap assembly, lithosphere
replacement, and processor mapping. The nearest depth node is selected, with
ties choosing the shallower node, and temperatures at nodes strictly above that
node are replaced. The supplied `exclude_buoy_*` settings remain independent
solver overrides; they are not recalculated from the temperature cutoff.

## Validation and resumption

Every expected source processor file must exist. During preparation the script
checks source headers, node counts, column counts, and finite values. It then
checks generated headers and every written value against the transformed data,
round-trips the generated input, and records SHA-256 hashes of all generated
input and IC files. Source size/mtime identities are checked for changes during
processing. This establishes input preparation correctness, not successful
completion of a subsequent CitcomS run.

Rerun the same command to resume. Matching completed folders are verified and
skipped; validated staging folders are reused. Incomplete or mismatched staging
folders are renamed with a `.previous-<timestamp>-<id>` suffix before rebuilding,
preserving their logs. Existing final folders without matching manifests, or
with changed generated files, stop the run and are never overwritten. Move
legacy serial outputs aside before preparing the same ages with this tool.

Changing `workers` or the requested age list does not invalidate an unchanged
resolved restart. Changes to the input template, effective restart settings,
mesh, code, or source file size/mtime do. Source identity checks use metadata,
not a full source-file checksum; use immutable master outputs during preparation.

On a worker failure, no more work is scheduled; running workers finish into
staging, and the command exits nonzero. Ctrl-C also lets active workers finish
into staging. It can therefore take time to return. A forced termination may
leave incomplete staging, which is handled on the next run. A filesystem lock
prevents simultaneous preparers in the same master directory. Atomic folder
renaming protects against process interruption; this is not a guarantee against
storage failure or sudden power loss.

## Generated solver settings

The IC prefix is `<model>` and the cycle suffix is `.0`; `datafile_old=<model>`,
`datadir_old=./ic_dir`, and `solution_cycles_init=0` agree with those files.
`restart=0` and `tic_method=-1` initialize the temperature from them. Structural
overrides conflicting with this layout are rejected. Other `CitcomS.*` overrides
are applied, including `chemical_buoyancy=0` from the example.

`RS_AGE` expands to the rounded folder age. The legacy `steps=RS_TIMESTEP`
default expands to the selected source timestep; `RS_TIMESTEP+2` expands to that
timestep plus two. These settings are carried over from the existing framework,
not a new choice of solver run duration. Set `CitcomS.steps` explicitly if your
solver workflow needs a different value.

## Checks and initial worker choice

```bash
python -W ignore::ResourceWarning -m unittest discover -s tests -p 'test_restart_citcoms_parallel.py' -v
```

The tests include real spawned workers, a decomposed mesh, comparison with the
serial temperature transformation, independent cutoff checks, failure/retry,
checksums, collision detection, folder protection, and ordered publication.
The warning filter suppresses existing file-handle warnings from Core modules.

Start with two workers and measure memory and wall time on a representative
subset. Each active age still loads a full model volume and constructs cap and
processor representations, so increasing workers also increases memory and I/O
demand. Full-model speedup has not been benchmarked by these synthetic checks.

Hidden workspaces and staging directories from earlier versions are ignored. They
are never scanned, migrated, or resumed. Configured model inputs and output paths
under hidden directories, including directory symlinks targeting hidden directories,
are rejected. Old hidden folders are left untouched; their disk usage is not cleaned
up automatically. Small hidden lock files remain for concurrency safety.

## Checkpoint setting in generated inputs

Dynamic-topography restarts generated by either workflow use `checkpointFrequency=0`
through the shared restart settings in `Core_Citcom.py`. Total-topography restarts
retain their original checkpoint default (`1`) and allow cfg overrides. This value is enforced after all inherited settings and cfg overrides. Both
generators check every active checkpoint assignment in the written file and
reject a missing or nonzero value. Existing generated inputs must be regenerated or edited;
the generator does not patch published restarts automatically.

This only changes generated inputs. The current C solver source writes an initial
checkpoint unconditionally and uses this value as a modulo divisor, so zero is
not a safe disable setting for an executable built from that source.
