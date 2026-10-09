"""Small real ASCII fixtures exercise the shared readers, writer and spawn workers."""

import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Core_Citcom
import Core_Util
import restart_citcoms
import restart_citcoms_parallel as parallel


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def create_model(root):
    Core_Util.verbose = Core_Citcom.verbose = False
    pid = parallel.flat_parameters(Core_Util.parse_configuration_file(str(ROOT / "sample_data/pid00000.cfg")))
    pid.update(nodex=3, nodey=3, nodez=5, nprocx=2, nprocy=1, nprocz=2,
               nproc_surf=1, coor=1, coor_file="data/0/model.coord.0", datafile="model",
               datadir="data/%RANK", start_age=165, output_format="ascii", marker_parameter="pid")
    pid["_SECTIONS_"] = []
    Core_Util.write_cfg_dictionary(pid, str(root / "master-pid.cfg"), True)
    template = copy.deepcopy(pid)
    template["marker_parameter"] = "template"
    Core_Util.write_cfg_dictionary(template, str(root / "master.cfg"), True)
    derived = Core_Citcom.derive_extra_citcom_parameters(pid)
    pid.update(derived)
    radii = [0.90, 0.94, 0.97, 0.99, 1.0]
    for rank in range(pid["total_proc"]):
        directory = root / "data" / str(rank)
        directory.mkdir(parents=True)
        with (directory / f"model.coord.{rank}").open("w") as stream:
            stream.write("  1" + str(pid["proc_node"]).rjust(8) + "\n")
            for yy in range(pid["ny"]):
                for xx in range(pid["nx"]):
                    for zz in range(pid["nz"]):
                        z = (rank % pid["nprocz"]) * (pid["nz"] - 1) + zz
                        stream.write(f"1.0 2.0 {radii[z]}\n")
        for step in (10, 20, 30):
            with (directory / f"model.velo.{rank}.{step}").open("w") as stream:
                stream.write(f"{step} {pid['proc_node']} 0.0\n1 {pid['proc_node']}\n")
                for yy in range(pid["ny"]):
                    for xx in range(pid["nx"]):
                        for zz in range(pid["nz"]):
                            # Shared nodes have identical values across processors.
                            z = (rank % 2) * (pid["nz"] - 1) + zz
                            x = (rank // 2) * (pid["nx"] - 1) + xx
                            value = (yy * 3 + x) * 5 + z
                            stream.write(f"{value}.0 {-value}.0 2.0 {0.5 + value / 1000 + step / 10000}\n")
    with (root / "model.time").open("w") as stream:
        for step, runtime in ((10, 0), (20, 5), (30, 10)):
            stream.write(f"{step} {runtime / pid['scalet']:.17g} 0.1 1 0.1\n")
    (root / "geodynamic_framework_defaults.conf").write_text("# Test defaults\ntest=0\n")
    config = """master_run_cfg = master.cfg
master_run_pid = master-pid.cfg
workers = 2
restart_ages = 155Ma, 165Ma, 160Ma, 165Ma
restart_type = dynamic_topography
restart_structure = separate
lithosphere_depth_DT = 100.0
lithosphere_temperature_DT = 0.3
CitcomS.solver.tsolver.finetunedt = 0.01
CitcomS.solver.tracer.chemical_buoyancy = 0
"""
    (root / "restart.cfg").write_text(config)


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        create_model(self.root)

    def cli(self):
        result = subprocess.run([sys.executable, str(ROOT / "restart_citcoms_parallel.py"), "restart.cfg"],
                                cwd=self.root, capture_output=True, text=True, timeout=60)
        return result

    def load(self):
        with working_directory(self.root), contextlib.redirect_stdout(io.StringIO()):
            return parallel.load_context("restart.cfg")

    def test_age_ranges_endpoints_and_invalid_steps(self):
        self.assertEqual(parallel.requested_ages("165Ma/0Ma/5Ma"), list(range(165, -1, -5)))
        self.assertEqual(parallel.requested_ages("0Ma/0Ma/5Ma"), [0])
        self.assertEqual(parallel.requested_ages("0Ma/10Ma/5Ma"), [0, 5, 10])
        for spec in ("10Ma/0Ma/0Ma", "10Ma/0Ma/-5Ma", "nanMa", "-1Ma"):
            with self.assertRaises(ValueError):
                parallel.requested_ages(spec)

    def test_serial_and_parallel_inputs_disable_checkpoint_output(self):
        context, tasks, _ = self.load()
        task = tasks[0]
        context["replacements"]["CitcomS.controller.checkpointFrequency"] = 100000
        self.assertEqual(parallel.build_input(context, task)["checkpointFrequency"], 0)
        serial = self.root / "serial-checkpoint-test"
        serial.mkdir()
        template = dict(context["template"])
        template.update({"_SECTIONS_": [], "datafile": "model", "datadir": "data/%RANK",
                         "coor_file": "coor.dat", "lith_age_file": "age.dat",
                         "slab_assim_file": "hist.dat", "checkpointFrequency": 100000})
        control = {"restart_type": "dynamic_topography", "rs_datafile": "model", "rs_datadir": "./ic_dir"}
        replacements = dict(Core_Citcom.dynamic_topography_restart_params)
        replacements["CitcomS.controller.checkpointFrequency"] = 100000
        with contextlib.redirect_stdout(io.StringIO()):
            generated = restart_citcoms.create_restart_run_cfg(
                template, control, replacements,
                str(serial), "test", task["age"], task["timestep"])
        self.assertEqual(generated["checkpointFrequency"], 0)
        with contextlib.redirect_stdout(io.StringIO()):
            parsed = Core_Util.parse_configuration_file(str(serial / "model_test.input"))
        self.assertEqual(parsed["checkpointFrequency"], 0)
        self.assertEqual(Core_Citcom.total_topography_restart_params["CitcomS.controller.checkpointFrequency"], 1)

    def test_checkpoint_final_check_rejects_missing_or_conflicting_values(self):
        path = self.root / "checkpoints.input"
        for text in ("", "checkpointFrequency=100000\n", "checkpointFrequency=0\n[CitcomS.controller]\ncheckpointFrequency=100000\n"):
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "checkpointFrequency=0"):
                Core_Citcom.check_restart_checkpoint_frequency(path)
        path.write_text("# checkpointFrequency=100000\ncheckpointFrequency=0\n")
        Core_Citcom.check_restart_checkpoint_frequency(path)
        parameters = {"checkpointFrequency": 100000, "CitcomS.controller": {"checkpointFrequency": 100000}}
        Core_Citcom.force_restart_checkpoint_frequency(parameters)
        self.assertEqual(parameters["checkpointFrequency"], 0)
        self.assertEqual(parameters["CitcomS.controller"]["checkpointFrequency"], 0)

    def test_coordinate_reference_comes_from_gridmaker_pid(self):
        template = self.root / "master.cfg"
        template.write_text(template.read_text().replace("coor_file=data/0/model.coord.0",
                                                         "coor_file=/missing/old-cluster/G5.coor.global.dat"))
        context, tasks, _ = self.load()
        generated = parallel.build_input(context, tasks[0])
        self.assertEqual(generated["coor_file"], "../data/0/model.coord.0")
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


    def test_spawn_matches_serial_transformation_and_resume(self):
        context, tasks, workers = self.load()
        self.assertEqual([task["age"] for task in tasks], [165, 160, 155])
        self.assertEqual(workers, 2)
        first = self.cli()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        publications = [line.split("Published ")[1] for line in first.stdout.splitlines() if "Published " in line]
        self.assertEqual(publications, [task["folder"] for task in tasks])
        # Compare actual generated files with the old temperature transformation.
        with working_directory(self.root), contextlib.redirect_stdout(io.StringIO()):
            master = Core_Citcom.get_all_pid_data("master-pid.cfg", verbose=False)
            for task in tasks:
                serial = self.root / f"serial-{task['age']}"
                serial.mkdir()
                control = {"lithosphere_depth_DT": 100, "lithosphere_temperature_DT": 0.3}
                restart_citcoms.create_no_lith_temp(control, master, {}, str(serial), "test", task["age"], task["timestep"])
                final = self.root / task["folder"]
                self.assertTrue(parallel.valid_manifest(final, context, task))
                for rank in range(context["pid"]["total_proc"]):
                    name = f"model.velo.{rank}.0"
                    expected = parallel.read_velocity(serial / "ic_dir" / name, context["pid"]["proc_node"])
                    actual = parallel.read_velocity(final / "ic_dir" / name, context["pid"]["proc_node"])
                    np.testing.assert_array_equal(actual, expected)
                    # Independent source-index calculation checks unchanged values
                    # and the cutoff (including the processor boundary node).
                    source = parallel.read_velocity(task["sources"][rank]["path"], context["pid"]["proc_node"])
                    np.testing.assert_array_equal(actual[:, :3], source[:, :3])
                    z = np.tile(np.arange(3) + (rank % 2) * 2, 6)
                    np.testing.assert_array_equal(actual[z <= 3, 3], source[z <= 3, 3])
                    np.testing.assert_array_equal(actual[z > 3, 3], np.full(np.sum(z > 3), 0.3))
                generated = Core_Util.parse_configuration_file(str(final / parallel.input_filename(context, task)))
                self.assertEqual(generated["datafile_old"], "model")
                self.assertEqual(generated["solution_cycles_init"], 0)
                self.assertEqual(generated["tic_method"], -1)
                self.assertEqual(generated["start_age"], task["age"])
                self.assertEqual(generated["steps"], task["timestep"])
                self.assertEqual(generated["chemical_buoyancy"], 0)
                self.assertEqual(generated["marker_parameter"], "template")
        before = {str(path): path.stat().st_mtime_ns for task in tasks
                  for path in (self.root / task["folder"]).rglob("*") if path.is_file()}
        # Changing concurrency alone must not invalidate finished outputs.
        cfg = self.root / "restart.cfg"
        cfg.write_text(cfg.read_text().replace("workers = 2", "workers = 1"))
        second = self.cli()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("skipping", second.stdout)
        self.assertEqual(before, {path: Path(path).stat().st_mtime_ns for path in before})

    def test_failed_age_is_not_published_and_retry_reuses_ready_age(self):
        source = self.root / "data/0/model.velo.0.10"
        original = source.read_text()
        source.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        failure = self.cli()
        self.assertEqual(failure.returncode, 1, failure.stdout + failure.stderr)
        self.assertFalse(list(self.root.glob("restart_dynamic_topography_*Ma")))
        self.assertTrue((self.root / "restart_dynamic_topography_165Ma.inprogress/failure.json").is_file())
        ready = self.root / "restart_dynamic_topography_160Ma.inprogress"
        self.assertTrue((ready / parallel.MANIFEST).is_file(), failure.stdout + failure.stderr)
        ready_time = (ready / parallel.MANIFEST).stat().st_mtime_ns
        source.write_text(original)
        success = self.cli()
        self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
        self.assertIn("Reusing validated staging", success.stdout)
        final_manifest = self.root / "restart_dynamic_topography_160Ma" / parallel.MANIFEST
        self.assertEqual(final_manifest.stat().st_mtime_ns, ready_time)
        self.assertTrue(list(self.root.glob("restart_dynamic_topography_165Ma.inprogress.previous-*")))

    def test_missing_input_preflight_and_existing_output_protection(self):
        missing = self.root / "data/1/model.velo.1.10"
        original = missing.read_bytes()
        missing.unlink()
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(list(self.root.glob("*.inprogress")))
        missing.write_bytes(original)
        final = self.root / "restart_dynamic_topography_165Ma"
        final.mkdir()
        (final / "important.txt").write_text("keep this")
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((final / "important.txt").read_text(), "keep this")

    def test_corrupt_finished_output_is_rejected(self):
        first = self.cli()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        output = self.root / "restart_dynamic_topography_165Ma/ic_dir/model.velo.0.0"
        lines = output.read_text().splitlines()
        lines[2] = lines[2].replace("2.0", "3.0")
        output.write_text("\n".join(lines) + "\n")
        second = self.cli()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("unverified", second.stdout)

    def test_sparse_outputs_skip_missing_requested_ages(self):
        context, _, _ = self.load()
        inventory = {10: str(self.root / "data/#/model.velo.#.10"),
                     20: str(self.root / "data/#/model.velo.#.20")}
        triples = [(10, 399.94, 0), (20, 389.96, 0)]
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            tasks = parallel.make_tasks([400, 395, 390, 385], triples, inventory, context["pid"])
        self.assertEqual([task["age"] for task in tasks], [400, 390])
        self.assertEqual([task["requested_ages"] for task in tasks], [[400], [390]])
        self.assertIn("Skipping requested 395 Ma", log.getvalue())
        self.assertIn("Skipping requested 385 Ma", log.getvalue())
        with self.assertRaisesRegex(ValueError, "No requested ages"), contextlib.redirect_stdout(io.StringIO()):
            parallel.make_tasks([395], triples, inventory, context["pid"])

    def test_initial_state_selects_exact_age_without_duplicate_restart(self):
        context, _, _ = self.load()
        inventory = {10: str(self.root / "data/#/model.velo.#.10"),
                     20: str(self.root / "data/#/model.velo.#.20")}
        tasks = parallel.make_tasks([400, 400], [(10, 400, 0), (20, 399.937, 0)], inventory, context["pid"])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["timestep"], 10)
        self.assertEqual(tasks[0]["folder"], "restart_dynamic_topography_400Ma")

    def test_folder_collision_is_disambiguated_and_invalid_workers(self):
        context, _, _ = self.load()
        inventory = {10: str(self.root / "data/#/model.velo.#.10"),
                     20: str(self.root / "data/#/model.velo.#.20")}
        triples = [(10, 165.1, 0), (20, 165.2, 0)]
        tasks = parallel.make_tasks([165.1, 165.2, 165.1], triples, inventory, context["pid"])
        self.assertEqual([task["timestep"] for task in tasks], [20, 10])
        self.assertEqual([task["folder"] for task in tasks],
                         ["restart_dynamic_topography_165Ma_step20", "restart_dynamic_topography_165Ma_step10"])
        self.assertEqual(tasks[1]["requested_ages"], [165.1, 165.1])
        reordered = parallel.make_tasks([165.2, 165.1], triples, inventory, context["pid"])
        self.assertEqual([task["folder"] for task in reordered], [task["folder"] for task in tasks])
        for task in tasks:
            with patch.object(parallel, "WORKER_CONTEXT", context):
                result = parallel.prepare_age(task)
            self.assertTrue(result["ok"], result)
            parallel.publish(context, task)
            self.assertTrue(parallel.valid_manifest(self.root / task["folder"], context, task))
        cfg = self.root / "restart.cfg"
        cfg.write_text(cfg.read_text().replace("workers = 2", "workers = 0"))
        with self.assertRaisesRegex(ValueError, "workers"):
            self.load()

    def test_cascade_holds_younger_age_until_oldest_is_ready(self):
        context, tasks, _ = self.load()
        oldest_started = threading.Event()
        younger_finished = threading.Event()
        submitted = []
        published = []
        def prepare(task):
            submitted.append(task["age"])
            if task["age"] == 165:
                oldest_started.set()
                self.assertTrue(younger_finished.wait(5))
                # The youngest must not start while this frontier is unfinished.
                self.assertNotIn(155, submitted)
            elif task["age"] == 160:
                self.assertTrue(oldest_started.wait(5))
                younger_finished.set()
            return {"ok": True, "seconds": 0}
        with patch.object(parallel, "prepare_age", side_effect=prepare), \
             patch.object(parallel, "publish", side_effect=lambda _, task: published.append(task["age"])), \
             contextlib.redirect_stdout(io.StringIO()):
            result = parallel.run_cascade(context, tasks, 2, lambda: ThreadPoolExecutor(max_workers=2))
        self.assertEqual(result, 0)
        self.assertEqual(published, [165, 160, 155])

    def test_same_directory_lock(self):
        with parallel.run_lock(self.root):
            with self.assertRaisesRegex(ValueError, "Another"):
                with parallel.run_lock(self.root):
                    pass

    def test_hidden_work_is_ignored_even_when_locked(self):
        context, tasks, _ = self.load()
        old = self.root / ".restart_dynamic_topography_165Ma.inprogress"
        old.mkdir(parents=True)
        (old / "retained.log").write_text("old work")
        with (old / ".prepare.lock").open("a+") as lease:
            parallel.fcntl.flock(lease, parallel.fcntl.LOCK_EX)
            with patch.object(parallel, "prepare_age", return_value={"ok": True, "seconds": 0}), \
                 patch.object(parallel, "publish"), contextlib.redirect_stdout(io.StringIO()):
                result = parallel.run_cascade(context, tasks, 1, lambda: ThreadPoolExecutor(max_workers=1))
        self.assertEqual(result, 0)
        self.assertEqual((old / "retained.log").read_text(), "old work")
        self.assertTrue(old.is_dir())

    def test_hidden_directories_and_symlink_targets_are_rejected(self):
        hidden = self.root / ".hidden-data"
        hidden.mkdir()
        alias = self.root / "visible-alias"
        alias.symlink_to(hidden, target_is_directory=True)
        for path in (hidden, alias):
            with self.assertRaisesRegex(ValueError, "Hidden directories"):
                parallel.visible_directory(path)
            with self.assertRaisesRegex(ValueError, "Hidden directories"):
                parallel.visible_file(path / "input.dat")

    def test_live_worker_staging_is_not_archived(self):
        stage = self.root / "restart_dynamic_topography_165Ma.inprogress"
        stage.mkdir()
        with (stage / ".prepare.lock").open("a+") as lease:
            parallel.fcntl.flock(lease, parallel.fcntl.LOCK_EX)
            with self.assertRaisesRegex(ValueError, "still preparing"):
                parallel.archive_stage(stage)
        self.assertTrue(stage.is_dir())

    def test_legacy_system_exit_becomes_a_failed_age(self):
        context, tasks, _ = self.load()
        with patch.object(parallel, "WORKER_CONTEXT", context), \
             patch.object(Core_Citcom, "read_proc_files_to_cap_list", side_effect=SystemExit("reader failed")):
            result = parallel.prepare_age(tasks[0])
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "reader failed")
        stage = parallel.staging_path(self.root, tasks[0])
        self.assertTrue((stage / "failure.json").is_file())
        self.assertFalse((stage / parallel.MANIFEST).exists())

    def test_one_and_two_workers_produce_identical_restart_files(self):
        multiple = self.cli()
        self.assertEqual(multiple.returncode, 0, multiple.stdout + multiple.stderr)
        with tempfile.TemporaryDirectory() as other:
            other_root = Path(other)
            create_model(other_root)
            cfg = other_root / "restart.cfg"
            cfg.write_text(cfg.read_text().replace("workers = 2", "workers = 1"))
            single = subprocess.run([sys.executable, str(ROOT / "restart_citcoms_parallel.py"), "restart.cfg"],
                                    cwd=other_root, capture_output=True, text=True, timeout=60)
            self.assertEqual(single.returncode, 0, single.stdout + single.stderr)
            for path in self.root.glob("restart_dynamic_topography_*Ma/**/*"):
                if path.is_file() and (path.suffix == ".input" or ".velo." in path.name):
                    self.assertEqual(path.read_bytes(), (other_root / path.relative_to(self.root)).read_bytes())

    def test_changed_settings_refuse_to_reuse_published_output(self):
        first = self.cli()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        cfg = self.root / "restart.cfg"
        cfg.write_text(cfg.read_text().replace("lithosphere_temperature_DT = 0.3", "lithosphere_temperature_DT = 0.4"))
        second = self.cli()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("does not match", second.stdout)

    def test_interrupt_stops_publication_and_waits_for_active_work(self):
        context, tasks, _ = self.load()
        finished = threading.Event()
        def prepare(task):
            finished.set()
            return {"ok": True, "seconds": 0}
        with patch.object(parallel, "prepare_age", side_effect=prepare), \
             patch.object(parallel, "wait", side_effect=KeyboardInterrupt), \
             patch.object(parallel, "publish") as published, \
             contextlib.redirect_stdout(io.StringIO()):
            result = parallel.run_cascade(context, tasks, 2, lambda: ThreadPoolExecutor(max_workers=2))
        self.assertEqual(result, 1)
        self.assertTrue(finished.is_set())
        published.assert_not_called()


if __name__ == "__main__":
    unittest.main()
