import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_script(script: Path, *args):
    cmd = [sys.executable, str(script), *map(str, args)]
    subprocess.check_call(cmd)


def test_compute_metric_cli(tmp_path):
    x = np.linspace(-1, 1, 21)
    n = np.exp(-x**2)
    v_hxc = np.zeros_like(n)
    npz_path = tmp_path / "data.npz"
    np.savez(npz_path, x=x, n=n, v_hxc=v_hxc)
    out_csv = tmp_path / "out.csv"
    metadata = {"system_id": "test", "spin": "u", "N_el": 1}
    run_script(
        Path("bin/compute_metric.py"),
        npz_path,
        out_csv,
        "--construction",
        "direct",
        "--metadata",
        json.dumps(metadata),
    )
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["system_id"] == "test"
    assert float(row["M_ratio"]) < 1e-10


def test_make_dataset_cli(tmp_path):
    out_csv = tmp_path / "dataset.csv"
    npz_dir = tmp_path / "npz"
    subprocess.check_call(
        [
            sys.executable,
            "bin/make_dataset.py",
            out_csv,
            "--npz-dir",
            npz_dir,
            "--k",
            "1.0",
            "--lambda",
            "20.0",
            "--grid",
            "51",
            "--spins",
            "u",
        ]
    )
    assert out_csv.exists()
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    (npz_file,) = npz_dir.iterdir()
    data = np.load(npz_file)
    assert "x" in data and "n" in data
