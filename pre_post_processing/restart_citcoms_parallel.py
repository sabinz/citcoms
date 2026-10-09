#!/usr/bin/env python3
"""Prepare dynamic-topography restart inputs in a resumable oldest-first cascade.

Run from the master model directory: python restart_citcoms_parallel.py restart.cfg
Concurrency is configured with ``workers = N`` in restart.cfg. No solver is run.
"""

import argparse
import contextlib
import copy
import datetime
import fcntl
import glob
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

import Core_Citcom
import Core_Util


MANIFEST = "restart_complete.json"
FORMAT_VERSION = 1
WORKER_CONTEXT = None
PATH_KEYS = ("coor_file", "lith_age_file", "slab_assim_file")


def report(message):
    print(f"{Core_Util.now()} {message}", flush=True)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    visible_file(path)
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def visible_directory(path):
    """Reject hidden model directories, including targets of directory symlinks."""
    path = Path(path).absolute()
    for candidate in (path, path.resolve()):
        if any(part.startswith(".") and part not in (".", "..") for part in candidate.parts):
            raise ValueError(f"Hidden directories are not used by this workflow: {path}")
    return path


def visible_file(path):
    path = Path(path).absolute()
    visible_directory(path.parent)
    visible_directory(path.resolve().parent)
    return path


def visible_pid(path):
    visible_file(path)
    settings = Core_Util.parse_configuration_file(str(path))
    for key in ("datadir", "coor_file"):
        if key in settings:
            value = str(settings[key]).strip("\"'")
            if key == "datadir":
                visible_directory(value.replace("%RANK", "0").replace("#", "0"))
            else:
                visible_file(value)
    if "datafile" in settings and "datadir" in settings:
        datafile = str(settings["datafile"])
        datadir = str(settings["datadir"]).strip("\"'").replace("%RANK", "0")
        visible_file(f"{datafile}.time")
        visible_file(Path(datadir) / f"{datafile}.time")
        visible_file(Path(datadir) / f"{datafile}.coord.0")


def file_identity(path):
    path = visible_file(path).resolve()
    stat = path.stat()
    if not path.is_file() or stat.st_size == 0:
        raise ValueError(f"Missing or empty input file: {path}")
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def flat_parameters(parsed):
    """Use the parser's effective top-level values; omit framework bookkeeping."""
    return {key: copy.deepcopy(value) for key, value in parsed.items()
            if not key.startswith(("_", "#")) and not isinstance(value, dict)}


def canonical_value(value):
    """Account for the framework parser's on/off and comma-list conversions."""
    if isinstance(value, (list, tuple)):
        return tuple(canonical_value(item) for item in value)
    if isinstance(value, str):
        if "," in value:
            return tuple(canonical_value(item.strip()) for item in value.split(","))
        if value.lower() in ("on", "true"):
            return 1
        if value.lower() in ("off", "false"):
            return 0
        try:
            return float(value)
        except ValueError:
            pass
        return value
    return value


def requested_ages(spec):
    """Parse Ma lists/ranges without the legacy parser's endpoint/zero-step traps."""
    def number(value):
        value = str(value).strip()
        if value.endswith("Ma"):
            value = value[:-2].strip()
        age = float(value)
        if not math.isfinite(age) or age < 0:
            raise ValueError("restart_ages must contain finite nonnegative ages in Ma")
        return age

    if isinstance(spec, (list, tuple)):
        ages = [number(value) for value in spec]
    else:
        text = str(spec).strip().strip("[]")
        if "/" in text:
            parts = text.split("/")
            if len(parts) != 3:
                raise ValueError("restart_ages range must be startMa/endMa/stepMa")
            start, end, step = map(number, parts)
            if step <= 0:
                raise ValueError("restart_ages step must be positive")
            count = int(math.floor(abs(end - start) / step + 1e-10)) + 1
            if count > 100000:
                raise ValueError("restart_ages requests more than 100000 ages")
            direction = 1 if end >= start else -1
            ages = [start + direction * index * step for index in range(count)]
        else:
            ages = [number(value) for value in text.split(",")]
    if not ages:
        raise ValueError("restart_ages cannot be empty")
    return ages


def source_inventory(pid):
    """Locate velocity files, including %RANK, ranked and flat directory layouts."""
    datafile = str(pid["datafile"])
    if Path(datafile).name != datafile or datafile in ("", ".", ".."):
        raise ValueError("datafile must be a filename prefix without directories")
    datadir = str(pid["datadir"]).strip('"\'')
    candidates = []
    if "%RANK" in datadir:
        candidates.append(datadir.replace("%RANK", "#"))
    else:
        candidates.extend([os.path.join(datadir, "#"), datadir])
    candidates.extend(["data/#", "Data/#"])
    for directory in dict.fromkeys(candidates):
        visible_directory(directory.replace("#", "0"))
        pattern = str(Path(directory.replace("#", "0")) / f"{datafile}.velo.0.*")
        steps = {}
        for filename in glob.glob(glob.escape(pattern[:-1]) + "*"):
            suffix = filename.rsplit(".", 1)[-1]
            if suffix.isdigit():
                steps[int(suffix)] = str(Path(directory).absolute() / f"{datafile}.velo.#.{suffix}")
        if steps:
            return steps
    raise ValueError(f"No processor-zero velocity files found for {datafile} in {datadir}")


def make_tasks(ages, triples, inventory, pid):
    available = [(int(step), float(age)) for step, age, _ in triples
                 if int(step) in inventory and math.isfinite(float(age))]
    if not available:
        raise ValueError("No velocity-file timesteps match the master time table")
    tasks = {}
    for requested in ages:
        # CitcomS outputs are slightly offset from nominal integer ages (e.g.
        # 389.962 Ma is the 390 Ma output). Never borrow a different age's output.
        label = int(np.around(requested))
        candidates = [item for item in available if int(np.around(item[1])) == label]
        if not candidates:
            report(f"Skipping requested {requested:g} Ma: no available velocity output rounding to {label} Ma")
            continue
        timestep, actual_age = min(candidates,
                                  key=lambda item: (abs(item[1] - requested), item[1], item[0]))
        age = int(np.around(actual_age))
        folder = f"restart_dynamic_topography_{age}Ma"
        if timestep in tasks:
            tasks[timestep]["requested_ages"].append(requested)
            continue
        pattern = inventory[timestep]
        names, _ = Core_Citcom.define_cap_or_proc_names(pid, pattern, "proc")
        sources = [file_identity(name) for name in names]
        tasks[timestep] = {"age": age, "actual_age": actual_age, "timestep": timestep,
                           "requested_ages": [requested], "folder": folder,
                           "source_pattern": pattern, "sources": sources}
    if not tasks:
        raise ValueError("No requested ages have matching available velocity outputs")
    # Disambiguate every member of a collision group, independent of request order.
    # Internal filenames may retain the rounded age: each restart has its own root.
    groups = {}
    for task in tasks.values():
        groups.setdefault(task["folder"], []).append(task)
    for folder, group in groups.items():
        if len(group) > 1:
            for task in group:
                task["folder"] = f"{folder}_step{task['timestep']}"
    return sorted(tasks.values(), key=lambda item: (-item["actual_age"], item["timestep"]))


def reference_exists(path, total_proc):
    visible_file(path)
    text = str(path).strip('"\'')
    if "%RANK" in text or "#" in text:
        return all(Path(text.replace("%RANK", str(rank)).replace("#", str(rank))).is_file()
                   for rank in range(total_proc))
    return Path(text).is_file() or bool(glob.glob(glob.escape(text) + ".*"))


def build_input(context, task):
    result = copy.deepcopy(context["template"])
    # The gridmaker PID describes the usable mesh on the current machine;
    # the original run template may retain a coordinate path from a cluster.
    # Explicit CitcomS.* overrides below still take precedence.
    if "coor_file" in context["pid"]:
        result["coor_file"] = context["pid"]["coor_file"]
    replacements = copy.deepcopy(context["replacements"])
    # Runtime inputs are flat, as with the existing Core_Util writer. Framework
    # section names on replacements specify their parameter, not solver syntax.
    for qualified, value in replacements.items():
        key = qualified.rsplit(".", 1)[-1]
        # These legacy COMMENT defaults were meant to remove section duplicates,
        # not the required active paths from a flat solver input.
        if value == "COMMENT" and key in PATH_KEYS + ("datadir", "datadir_old"):
            continue
        if value in ("DELETE", "COMMENT"):
            result.pop(key, None)
        elif value == "RS_TIMESTEP":
            result[key] = 0 if key == "solution_cycles_init" else task["timestep"]
        elif value == "RS_TIMESTEP+2":
            result[key] = task["timestep"] + 2
        elif value == "RS_AGE":
            result[key] = task["age"]
        else:
            result[key] = value

    required = {"datafile": context["pid"]["datafile"],
                "datadir": f"./Age{task['age']}Ma/%RANK",
                "datafile_old": context["pid"]["datafile"], "datadir_old": "./ic_dir",
                "start_age": task["age"], "solution_cycles_init": 0,
                "restart": 0, "tic_method": -1}
    for key, expected in required.items():
        for qualified, value in context["overrides"].items():
            if qualified.rsplit(".", 1)[-1] == key and value != expected:
                raise ValueError(f"Override {qualified}={value} conflicts with generated restart layout ({expected})")
        result[key] = expected
    for key in PATH_KEYS:
        if key in result:
            original = str(result[key]).strip('"\'')
            if not os.path.isabs(original):
                original = os.path.normpath(os.path.join("..", original))
            result[key] = original
    # Do not serialize framework sections or derived gridmaker parameters.
    Core_Citcom.force_restart_checkpoint_frequency(result)
    result["_SECTIONS_"] = []
    if any(isinstance(value, str) and value.startswith("RS_") for value in result.values()):
        raise ValueError("Unresolved restart placeholder in generated input")
    return result


def input_filename(context, task):
    return f"{context['pid']['datafile']}_dynamic_topography_{task['age']}Ma.input"


def task_signature(context, task):
    # Requested ages and worker count do not change the resolved restart contents.
    return digest_json({"format": FORMAT_VERSION, "context": context["signature"],
                        "age": task["actual_age"], "timestep": task["timestep"],
                        "sources": task["sources"], "input": build_input(context, task)})


def check_sources(task):
    for expected in task["sources"]:
        if file_identity(expected["path"]) != expected:
            raise ValueError(f"Source changed during processing: {expected['path']}")


def read_velocity(path, rows, expected_headers=None):
    with open(path) as stream:
        headers = [stream.readline().rstrip("\n") for _ in range(2)]
        if any(not header.strip() for header in headers):
            raise ValueError(f"Missing velocity headers: {path}")
        try:
            first = headers[0].split()
            second = headers[1].split()
            if (len(first) != 3 or len(second) != 2 or int(first[1]) != rows
                    or int(second[1]) != rows or int(second[0]) != 1
                    or not math.isfinite(float(first[2]))):
                raise ValueError("invalid node counts or header fields")
            int(first[0])
        except (ValueError, IndexError) as error:
            raise ValueError(f"Invalid velocity header in {path}: {error}") from error
        if expected_headers is not None and headers != list(expected_headers):
            raise ValueError(f"Unexpected restart header: {path}")
        data = np.loadtxt(stream, ndmin=2)
    if data.shape[0] != rows or data.shape[1] < 4 or not np.isfinite(data).all():
        raise ValueError(f"Invalid velocity data in {path}: shape={data.shape}, expected {rows} rows and >=4 finite columns")
    return data


def valid_manifest(directory, context, task):
    """A folder is reusable only if its signature and all generated bytes agree."""
    directory = visible_directory(directory)
    try:
        manifest = json.loads((directory / MANIFEST).read_text())
        if manifest["signature"] != task_signature(context, task):
            return False
        expected_names = {f"ic_dir/{context['pid']['datafile']}.velo.{rank}.0"
                          for rank in range(context["pid"]["total_proc"])}
        expected_names.add(input_filename(context, task))
        if set(manifest["files"]) != expected_names:
            return False
        visible_directory(directory / "ic_dir")
        if not visible_directory(directory / f"Age{task['age']}Ma").is_dir():
            return False
        for relative, record in manifest["files"].items():
            path = directory / relative
            if path.is_symlink() or path.stat().st_size != record["size"] or file_digest(path) != record["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def staging_path(root, task):
    return visible_directory(Path(root) / (task["folder"] + ".inprogress"))


def initialize_worker(context):
    global WORKER_CONTEXT
    WORKER_CONTEXT = context
    # Parent stops scheduling on Ctrl-C; active ages finish into safe staging.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    Core_Util.verbose = False
    Core_Citcom.verbose = False


def prepare_age(task):
    context = WORKER_CONTEXT
    stage = staging_path(context["root"], task)
    log_path = stage / "prepare.log"
    started = time.monotonic()
    stage.mkdir(exist_ok=False)
    with (stage / ".prepare.lock").open("a+") as lease, \
            log_path.open("w", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            report(f"Preparing {task['actual_age']} Ma, source timestep {task['timestep']}")
            check_sources(task)
            pid = context["pid"]
            # Unlike the shared reader, explicitly reject missing/truncated input
            # instead of allowing zero padding for volume data.
            for source in task["sources"]:
                read_velocity(source["path"], pid["proc_node"])
            data_by_cap = Core_Citcom.read_proc_files_to_cap_list(pid, task["source_pattern"], "temp")
            mask = np.tile(np.arange(pid["nodez"]) > context["znode"], pid["nodex"] * pid["nodey"])
            for cap, data in enumerate(data_by_cap):
                array = np.asarray(data, dtype=float)
                if array.shape[0] != len(mask) or array.shape[1] < 4 or not np.isfinite(array).all():
                    raise ValueError(f"Invalid assembled cap {cap}")
                array[mask, 3] = context["temperature"]
                data_by_cap[cap] = array.tolist()
            out_data = Core_Citcom.get_proc_list_from_cap_list(pid, data_by_cap)
            del data_by_cap
            (stage / "ic_dir").mkdir()
            (stage / f"Age{task['age']}Ma").mkdir()
            pattern = str(stage / "ic_dir" / f"{pid['datafile']}.velo.#.0")
            names = Core_Citcom.write_cap_or_proc_list_to_files(pid, pattern, (out_data,), "proc", True)
            _, headers = Core_Citcom.define_cap_or_proc_names(pid, pattern, "proc")
            if len(names) != pid["total_proc"]:
                raise ValueError("Writer returned an incomplete processor file set")
            records = {}
            for rank, name in enumerate(names):
                written = read_velocity(name, pid["proc_node"], headers)
                if not np.array_equal(written, np.asarray(out_data[rank])):
                    raise ValueError(f"Restart values differ from transformed data: {name}")
                relative = str(Path(name).relative_to(stage))
                records[relative] = {"size": Path(name).stat().st_size, "sha256": file_digest(name)}
            del out_data
            config_path = stage / input_filename(context, task)
            expected = build_input(context, task)
            Core_Util.write_cfg_dictionary(expected, str(config_path), True)
            Core_Citcom.check_restart_checkpoint_frequency(config_path)
            parsed = flat_parameters(Core_Util.parse_configuration_file(str(config_path)))
            wanted = flat_parameters(expected)
            if set(parsed) != set(wanted):
                raise ValueError("Generated input did not round-trip all parameters")
            for key in wanted:
                if canonical_value(parsed[key]) != canonical_value(wanted[key]):
                    raise ValueError(f"Generated input has incorrect {key}: {parsed[key]}")
            os.chmod(config_path, 0o775)
            records[config_path.name] = {"size": config_path.stat().st_size, "sha256": file_digest(config_path)}
            check_sources(task)
            write_json(stage / MANIFEST, {"format": FORMAT_VERSION, "signature": task_signature(context, task),
                       "source_timestep": task["timestep"], "actual_age": task["actual_age"],
                       "folder_age": task["age"], "sources": task["sources"],
                       "znode": context["znode"], "temperature": context["temperature"],
                       "files": records, "validated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            report("Validated; awaiting oldest-first publication")
            return {"ok": True, "seconds": time.monotonic() - started}
        except (Exception, SystemExit) as error:
            traceback.print_exc()
            write_json(stage / "failure.json", {"error": str(error), "timestep": task["timestep"]})
            return {"ok": False, "error": str(error), "seconds": time.monotonic() - started}


def archive_stage(stage):
    suffix = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    archive = stage.with_name(stage.name + ".previous-" + suffix)
    # A forcibly terminated parent can leave live children finishing an age.
    # Never move their working directory out from underneath them.
    with (stage / ".prepare.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"An earlier worker is still preparing {stage}; wait for it to finish before resuming") from error
        stage.rename(archive)
    report(f"Preserved incomplete/mismatched staging folder: {archive.name}")


def publish(context, task):
    stage = staging_path(context["root"], task)
    final = Path(context["root"]) / task["folder"]
    check_sources(task)
    if not valid_manifest(stage, context, task):
        raise ValueError(f"Staging validation failed for {stage}")
    if final.exists():
        raise ValueError(f"Refusing to replace existing folder: {final}")
    # Sibling directories share a filesystem. The root lock excludes other
    # instances of this tool; the final name becomes visible in one rename.
    stage.rename(final)
    report(f"Published {final.name}")


def run_cascade(context, tasks, workers, executor_factory=None):
    root = Path(context["root"])
    states = [None] * len(tasks)
    # Validate all existing final folders before starting any new work.
    for index, task in enumerate(tasks):
        final = visible_directory(root / task["folder"])
        if final.exists():
            if not valid_manifest(final, context, task):
                raise ValueError(f"Existing folder is unverified or does not match current inputs: {final}. "
                                 "Move it aside or use a separate master-run directory; it will not be overwritten.")
            states[index] = "published"
            report(f"Verified existing {final.name}; skipping")
        else:
            stage = staging_path(root, task)
            if stage.exists():
                if valid_manifest(stage, context, task):
                    states[index] = "ready"
                    report(f"Reusing validated staging folder for {task['age']} Ma")
                else:
                    archive_stage(stage)
    if executor_factory is None:
        executor_factory = lambda: ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker, initargs=(context,))
    executor = executor_factory()
    futures = {}
    frontier = 0
    failed = False
    try:
        while frontier < len(tasks):
            while frontier < len(tasks) and states[frontier] in ("ready", "published"):
                if states[frontier] == "ready":
                    publish(context, tasks[frontier])
                    states[frontier] = "published"
                frontier += 1
            if frontier == len(tasks):
                break
            # No more than workers ages ahead of the publication frontier.
            for index in range(frontier, min(len(tasks), frontier + workers)):
                if states[index] is None:
                    report(f"Starting {tasks[index]['age']} Ma (source timestep {tasks[index]['timestep']})")
                    futures[executor.submit(prepare_age, tasks[index])] = index
                    states[index] = "running"
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                index = futures.pop(future)
                try:
                    result = future.result()
                except (Exception, SystemExit) as error:
                    result = {"ok": False, "error": str(error)}
                states[index] = "ready" if result["ok"] else "failed"
                if result["ok"]:
                    report(f"Validated {tasks[index]['age']} Ma in {result['seconds']:.1f}s")
                else:
                    failed = True
                    report(f"FAILED {tasks[index]['age']} Ma: {result['error']}; see "
                           f"{staging_path(root, tasks[index]) / 'prepare.log'}")
            if failed:
                report("Stopping the cascade; active workers will finish into staging for the next run")
                break
    except KeyboardInterrupt:
        report("Interrupted; active workers will finish into staging. Rerun the same configuration to resume.")
        failed = True
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True)
    report(f"{states.count('published')}/{len(tasks)} ages published; "
           "completed staging folders are reusable on the next run")
    return 1 if failed else 0


@contextlib.contextmanager
def run_lock(root):
    # flock releases even after a crash; the persistent filename is not a stale lock.
    visible_directory(root)
    with (Path(root) / ".restart_citcoms_parallel.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another parallel restart preparer is using this model directory") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_context(config_path):
    Core_Util.verbose = False
    Core_Citcom.verbose = False
    visible_directory(Path.cwd())
    visible_file(config_path)
    control = Core_Util.parse_configuration_file(str(config_path))
    if control.get("restart_type") != "dynamic_topography" or control.get("restart_structure") != "separate":
        raise ValueError("This script requires restart_type=dynamic_topography and restart_structure=separate")
    workers = control.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer in the configuration file")
    for key in ("master_run_cfg", "master_run_pid", "restart_ages", "lithosphere_depth_DT", "lithosphere_temperature_DT"):
        if key not in control:
            raise ValueError(f"Missing required setting: {key}")
    ages = requested_ages(control["restart_ages"])
    temperature = float(control["lithosphere_temperature_DT"])
    depth = float(control["lithosphere_depth_DT"])
    if not math.isfinite(temperature) or not 0 <= temperature <= 1 or not math.isfinite(depth):
        raise ValueError("Temperature must be in [0,1] and lithosphere depth must be finite")
    visible_file(control["master_run_cfg"])
    visible_pid(control["master_run_pid"])
    master = Core_Citcom.get_all_pid_data(control["master_run_pid"], verbose=False)
    pid = master["pid_d"]
    if pid.get("output_format") != "ascii":
        raise ValueError("Only ASCII processor velocity files are supported")
    if not master.get("time_d") or not master.get("coor_d"):
        raise ValueError("Master time table and depth coordinates are required")
    for nodes, procs in (("nodex", "nprocx"), ("nodey", "nprocy"), ("nodez", "nprocz")):
        if pid[procs] < 1 or pid[nodes] < 2 or (pid[nodes] - 1) % pid[procs]:
            raise ValueError(f"Invalid mesh decomposition: {nodes}/{procs}")
    depths = master["coor_d"]["depth_km"]
    if len(depths) != pid["nodez"] or not np.isfinite(depths).all() or not min(depths) <= depth <= max(depths):
        raise ValueError("Lithosphere depth is outside the mesh, or depth coordinates are invalid")
    # Same nearest-node/tie-to-shallower rule as the legacy helper, including
    # the deepest endpoint that the legacy implementation cannot handle.
    znode = min(range(len(depths)), key=lambda index: (abs(depths[index] - depth), -index))
    template = flat_parameters(Core_Util.parse_configuration_file(control["master_run_cfg"]))
    for key in ("nodex", "nodey", "nodez", "nproc_surf", "nprocx", "nprocy", "nprocz"):
        if key in template and template[key] != pid[key]:
            raise ValueError(f"master_run_cfg and master_run_pid disagree on {key}")
    overrides = {key: value for key, value in control.items() if key.startswith("CitcomS.")}
    replacements = copy.deepcopy(Core_Citcom.dynamic_topography_restart_params)
    replacements.update(overrides)
    context = {"root": str(Path.cwd()), "pid": pid, "template": template,
               "replacements": replacements, "overrides": overrides,
               "temperature": temperature, "znode": znode}
    context["signature"] = digest_json({"pid": pid, "template": template,
        "replacements": replacements, "temperature": temperature, "znode": znode,
        "depths": depths, "code": {Path(path).name: file_digest(path)
             for path in (__file__, Core_Citcom.__file__, Core_Util.__file__)}})
    tasks = make_tasks(ages, master["time_d"]["triples"], source_inventory(pid), pid)
    # Check active external references against the final folder's location.
    for task in tasks:
        generated = build_input(context, task)
        active = {"coor_file": generated.get("coor", 0),
                  "lith_age_file": generated.get("lith_age", 0),
                  "slab_assim_file": generated.get("slab_assim", 0)}
        for key, enabled in active.items():
            if enabled not in (False, 0, "0", "off", "False", None):
                if key not in generated:
                    raise ValueError(f"Missing active reference {key}")
                reference = Path(context["root"]) / task["folder"] / generated[key]
                # Normalize before testing: the final directory does not exist yet.
                if not reference_exists(os.path.normpath(reference), pid["total_proc"]):
                    raise ValueError(f"Missing active reference {key}: {reference}")
    return context, tasks, workers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", nargs="?", help="restart configuration; paths relative to master-run cwd")
    parser.add_argument("-e", action="store_true", help="print example configuration")
    args = parser.parse_args(argv)
    if args.e:
        print((Path(__file__).parent / "sample_data" / "restart_citcoms_parallel.cfg").read_text(), end="")
        return 0
    if not args.configuration:
        parser.error("a configuration file is required (or use -e)")
    try:
        with run_lock(Path.cwd()):
            context, tasks, workers = load_context(args.configuration)
            report(f"Preparing {len(tasks)} distinct ages oldest-first with up to {workers} workers")
            report(f"Lithosphere cutoff: znode {context['znode']}; temperature {context['temperature']}")
            for task in tasks:
                report(f"Requested {task['requested_ages']} Ma -> {task['actual_age']:.6g} Ma, "
                       f"timestep {task['timestep']}, folder {task['folder']}")
            return run_cascade(context, tasks, workers)
    except Exception as error:
        report(f"ERROR: {error}")
        return 1
    except KeyboardInterrupt:
        report("Interrupted before processing started")
        return 130


if __name__ == "__main__":
    sys.exit(main())
