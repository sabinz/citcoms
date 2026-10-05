"""Cascade tests with tiny ASCII models and real GMT where available."""

import contextlib
import io
import json
import math
import os
from pathlib import Path
import shutil
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
import grid_maker_parallel as cascade
from test_restart_citcoms_parallel import create_model, working_directory


class GridTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        create_model(self.root)
        pid_file = self.root / "master-pid.cfg"
        pid = {key: value for key, value in Core_Util.parse_configuration_file(str(pid_file)).items()
               if not isinstance(value, dict) and not key.startswith("_")}
        pid.update(lon_min=30, lon_max=50, lat_min=40, lat_max=60)
        pid["_SECTIONS_"] = []
        Core_Util.write_cfg_dictionary(pid, str(pid_file), True)
        for rank in range(4):
            coor_file = self.root / f"data/{rank}/model.coord.{rank}"
            old = coor_file.read_text().splitlines()
            lines = [old[0]]
            for yy in range(3):
                for xx in range(2):
                    x = (rank // 2) + xx
                    for zz in range(3):
                        radius = old[len(lines)].split()[2]
                        lines.append(f"{math.radians(30 + 10 * yy)} {math.radians(30 + 10 * x)} {radius}")
            coor_file.write_text("\n".join(lines) + "\n")
            for step in (10, 20, 30):
                source = self.root / f"data/{rank}/model.velo.{rank}.{step}"
                headers = source.read_text().splitlines()[:2]
                data = []
                viscosity = []
                for yy in range(3):
                    for xx in range(2):
                        for zz in range(3):
                            z = (rank % 2) * 2 + zz
                            data.append(f"4 3 2 {0.5 + z * 0.1 + step * 0.001}")
                            viscosity.append(str(10 ** z))
                source.write_text("\n".join(headers + data) + "\n")
                (self.root / f"data/{rank}/model.visc.{rank}.{step}").write_text("header\n" + "\n".join(viscosity) + "\n")
        (self.root / "grid.cfg").write_text("""pid_file = master-pid.cfg
procs = 2
time_spec = 30, 10, 20, 10
level_spec = 0, 4
debug = False
shift_lon = False
[Temperature]
field = temp
dimensional = True
deviation = True
blockmedian_I = 5
surface_I = 5
T = 0
[Speed]
field = horiz_vmag
blockmedian_I = 5
surface_I = 5
""")

    def load(self):
        with working_directory(self.root), contextlib.redirect_stdout(io.StringIO()):
            return cascade.load("grid.cfg")

    def cli(self, root=None):
        return subprocess.run([sys.executable, str(ROOT / "grid_maker_parallel.py"), "grid.cfg"],
                              cwd=root or self.root, capture_output=True, text=True, timeout=90)

    def values(self, path):
        result = subprocess.run(["gmt", "grd2xyz", str(path)], capture_output=True, text=True, check=True)
        return np.loadtxt(io.StringIO(result.stdout))[:, 2]

    def test_time_and_level_requests(self):
        self.assertEqual(cascade.parse_times("165Ma/0Ma/5Ma")[0], (165, "Ma"))
        self.assertEqual(cascade.parse_times("0Ma/0Ma/5Ma"), [(0, "Ma")])
        self.assertEqual(cascade.parse_times("10,20Ma,5Myr"), [(10, "step"), (20, "Ma"), (5, "Myr")])
        self.assertEqual(cascade.parse_levels("4/0/2", 5), [4, 2, 0])
        requests = self.root / "requests.dat"
        requests.write_text("# Requests\n10\n20Ma\n5Myr\n")
        self.assertEqual(cascade.parse_times(str(requests)), [(10, "step"), (20, "Ma"), (5, "Myr")])
        for levels in ("-1", "5", "0.5", "4/0/0"):
            with self.assertRaises(ValueError):
                cascade.parse_levels(levels, 5)
        context, tasks, _ = self.load()
        self.assertEqual([task["timestep"] for task in tasks], [10, 20, 30])
        self.assertEqual([task["age"] for task in tasks], [165, 160, 155])

    def test_procs_cfg_preserved_and_takes_precedence(self):
        cfg = self.root / "grid.cfg"
        cfg.write_text(cfg.read_text().replace("procs = 2", "procs = 20\nverbose = off\nworkers = 4\ncores = 5"))
        context, _, procs = self.load()
        self.assertEqual(procs, 20)
        self.assertFalse(context["initialized"][0]["verbose"])
        with working_directory(self.root), contextlib.redirect_stdout(io.StringIO()):
            _, _, procs = cascade.load("grid.cfg", 1)
        self.assertEqual(procs, 1)

    def test_missing_input_and_filename_collisions(self):
        missing = self.root / "data/1/model.velo.1.10"
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()
        # Distinct levels truncating to the same km filename must be rejected.
        control = {"_SECTIONS_": ["A"], "A": {"field": "temp"}}
        with self.assertRaisesRegex(ValueError, "collision"):
            cascade.expected_outputs(control, {"datafile": "model"}, [0.1, 0.9], [0, 1], 165)

    def test_zero_byte_output_cannot_be_published(self):
        context, tasks, _ = self.load()
        task = tasks[0]
        wanted = cascade.signature(context, task)
        for category, names in task["outputs"].items():
            stage = cascade.folder(context, task, category, True)
            stage.mkdir(parents=True)
            for name in names:
                (stage / name).write_bytes(b"grid")
            cascade.write_json(stage / cascade.MANIFEST, {"signature": wanted,
                               "files": {name: 4 for name in names}})
        category = next(iter(task["outputs"]))
        (cascade.folder(context, task, category, True) / task["outputs"][category][0]).write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Missing, empty"):
            cascade.publish(context, task)
        self.assertFalse(any(cascade.folder(context, task, category).exists() for category in task["outputs"]))

    def test_partial_publication_resumes_remaining_field_renames(self):
        context, tasks, _ = self.load()
        task = tasks[0]
        wanted = cascade.signature(context, task)
        for category, names in task["outputs"].items():
            stage = cascade.folder(context, task, category, True)
            stage.mkdir(parents=True)
            for name in names:
                (stage / name).write_bytes(b"grid")
            cascade.write_json(stage / cascade.MANIFEST, {"signature": wanted, "files": {name: 4 for name in names}})
        category = next(iter(task["outputs"]))
        cascade.folder(context, task, category, True).rename(cascade.folder(context, task, category))
        self.assertEqual(cascade.inspect_age(context, task), "ready")
        with contextlib.redirect_stdout(io.StringIO()):
            cascade.publish(context, task)
        for category, names in task["outputs"].items():
            self.assertTrue(cascade.matching_folder(cascade.folder(context, task, category), names, wanted))

    def test_cascade_is_bounded_and_publishes_oldest_first(self):
        context, tasks, _ = self.load()
        submitted, published = [], []
        younger_finished = threading.Event()
        def prepare(task):
            submitted.append(task["age"])
            if task["age"] == 165:
                self.assertTrue(younger_finished.wait(5))
                self.assertNotIn(155, submitted)
            elif task["age"] == 160:
                younger_finished.set()
            return {"ok": True, "seconds": 0}
        with patch.object(cascade, "prepare", side_effect=prepare), \
             patch.object(cascade, "publish", side_effect=lambda _, task: published.append(task["age"])), \
             contextlib.redirect_stdout(io.StringIO()):
            result = cascade.cascade(context, tasks, 2, lambda: ThreadPoolExecutor(max_workers=2))
        self.assertEqual(result, 0)
        self.assertEqual(published, [165, 160, 155])

    def test_worker_failure_is_nonzero_and_stops_younger_scheduling(self):
        context, tasks, _ = self.load()
        submitted = []
        def prepare(task):
            submitted.append(task["age"])
            return {"ok": False, "error": "failed"}
        with patch.object(cascade, "prepare", side_effect=prepare), \
             patch.object(cascade, "publish") as published, \
             contextlib.redirect_stdout(io.StringIO()):
            result = cascade.cascade(context, tasks, 1, lambda: ThreadPoolExecutor(max_workers=1))
        self.assertEqual(result, 1)
        self.assertEqual(submitted, [165])
        published.assert_not_called()

    def test_false_flags_and_stale_coordinate_cache(self):
        (self.root / "_cache_lon_coords.txt").write_text("999\n")
        (self.root / "_cache_lat_coords.txt").write_text("999\n")
        context, tasks, _ = self.load()
        self.assertEqual(len(context["lon"]), 9)
        self.assertLess(context["lon"].max(), 100)
        self.assertFalse(context["initialized"][0]["debug"])
        self.assertFalse(any(name.endswith(".png") for names in tasks[0]["outputs"].values() for name in names))

    def test_hidden_workspace_is_migrated_to_visible_name(self):
        context, tasks, _ = self.load()
        old = self.root / "model/.gridmaker-work"
        age = old / "165Ma.inprogress"
        age.mkdir(parents=True)
        (age / "grid.log").write_text("retained log")
        with (age / ".worker.lock").open("a+") as lease:
            cascade.fcntl.flock(lease, cascade.fcntl.LOCK_EX)
            with self.assertRaisesRegex(ValueError, "still preparing"):
                cascade.make_workspace_visible(context)
        with contextlib.redirect_stdout(io.StringIO()):
            cascade.make_workspace_visible(context)
        visible = cascade.workspace(context, tasks[0])
        self.assertFalse(visible.parent.name.startswith("."))
        self.assertFalse(old.exists())
        self.assertEqual((visible / "grid.log").read_text(), "retained log")

    @unittest.skipUnless(shutil.which("gmt"), "GMT is not installed")
    def test_real_gmt_spawn_numerics_resume_and_zero_byte_repair(self):
        context, tasks, _ = self.load()
        first = self.cli()
        if first.returncode:
            logs = "\n".join(path.read_text() for path in (self.root / "model/gridmaker-work").rglob("grid.log"))
            self.fail(first.stdout + first.stderr + logs)
        publication = [line.split("Published all fields for ")[1] for line in first.stdout.splitlines() if "Published all fields for " in line]
        self.assertEqual(publication, ["165 Ma", "160 Ma", "155 Ma"])
        for task in tasks:
            for category, names in task["outputs"].items():
                final = cascade.folder(context, task, category)
                self.assertTrue(cascade.checked_files(final, names))
            for level in (0, 4):
                depth = int(context["initialized"][3]["depth_km"][level])
                expected = 0.5 + level * 0.1 + task["timestep"] * 0.001
                base = f"model_temp_t{task['age']}_{depth}.nc"
                np.testing.assert_allclose(self.values(cascade.folder(context, task, "temp") / base), expected, atol=1e-6)
                dimensional = base.replace(".nc", "_dimensional.nc")
                rule = context["dimension_map"]["temp"]
                scaled = expected * float(f"{rule['coef']:.6f}") + rule["const"]
                np.testing.assert_allclose(self.values(cascade.folder(context, task, "temp_dimensional") / dimensional),
                                           scaled, rtol=1e-5, atol=1e-5)
                speed = f"model_horiz_vmag_t{task['age']}_{depth}.nc"
                np.testing.assert_allclose(self.values(cascade.folder(context, task, "horiz_vmag") / speed), 5, atol=1e-6)
                deviation = "deviation_" + base
                np.testing.assert_allclose(self.values(cascade.folder(context, task, "temp_deviation") / deviation), 0, atol=1e-6)
        second = self.cli()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("Skipping completed", second.stdout)
        target = cascade.folder(context, tasks[0], "temp") / tasks[0]["outputs"]["temp"][0]
        target.write_bytes(b"")
        repair = self.cli()
        self.assertEqual(repair.returncode, 0, repair.stdout + repair.stderr)
        self.assertGreater(target.stat().st_size, 0)
        self.assertTrue(list(target.parent.parent.glob("165.previous-*")))

    @unittest.skipUnless(shutil.which("gmt"), "GMT is not installed")
    def test_real_gmt_one_and_two_workers_agree(self):
        first = self.cli()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        with tempfile.TemporaryDirectory() as other:
            other_root = Path(other)
            shutil.copytree(self.root, other_root, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("model", ".grid_maker_parallel.lock"))
            cfg = other_root / "grid.cfg"
            cfg.write_text(cfg.read_text().replace("procs = 2", "procs = 1"))
            second = self.cli(other_root)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            for path in (self.root / "model").rglob("*.nc"):
                if ".previous-" not in str(path):
                    np.testing.assert_allclose(self.values(path), self.values(other_root / path.relative_to(self.root)), atol=1e-6)

    def test_surface_files_are_required_only_on_top_processors(self):
        cfg = self.root / "grid.cfg"
        cfg.write_text("""pid_file = master-pid.cfg
procs = 1
time_spec = 10
level_spec = 4
[Surface]
field = surf_topography
""")
        for rank in (1, 3):
            (self.root / f"data/{rank}/model.surf.{rank}.10").write_text("header\n" + "0 1 2 3\n" * 6)
        context, tasks, workers = self.load()
        self.assertEqual(workers, 1)
        self.assertEqual(len(tasks[0]["sources"]), 2)
        (self.root / "data/3/model.surf.3.10").unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_changed_settings_do_not_reuse_manifests(self):
        context, tasks, _ = self.load()
        task = tasks[0]
        for category, names in task["outputs"].items():
            final = cascade.folder(context, task, category)
            final.mkdir(parents=True)
            for name in names:
                (final / name).write_bytes(b"grid")
            cascade.write_json(final / cascade.MANIFEST, {"signature": "outdated", "files": {name: 4 for name in names}})
        with self.assertRaisesRegex(ValueError, "different settings"):
            cascade.inspect_age(context, task)


if __name__ == "__main__":
    unittest.main()
