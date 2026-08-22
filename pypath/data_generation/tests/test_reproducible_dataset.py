#!/usr/bin/env python3
"""无需原生二进制即可执行的数据生成与契约回归检查。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable


class ReproducibleDatasetTest(unittest.TestCase):
    def test_skip_native_generation_and_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wmpc-repro-test-", dir=REPO_ROOT) as raw:
            output = Path(raw) / "generated"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            subprocess.run(
                [
                    PYTHON,
                    "pypath/data_generation/generate_reproducible_dataset.py",
                    "--output-dir",
                    str(output),
                    "--num-circuits",
                    "2",
                    "--node-count",
                    "3",
                    "--seed",
                    "13",
                    "--skip-native",
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [
                    PYTHON,
                    "pypath/data_generation/validate_reproducible_dataset.py",
                    "--root",
                    str(output),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            self.assertTrue(report["valid"])
            self.assertEqual(report["circuit_count"], 2)
            self.assertEqual(report["netlist_count"], 2)
            self.assertEqual(report["trajectory_count"], 0)
            self.assertEqual(report["warmup_count"], 0)


if __name__ == "__main__":
    unittest.main()
