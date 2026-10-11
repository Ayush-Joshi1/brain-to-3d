"""Reproduce official validation metrics for the selected generic medoid."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "train.npz").is_file() and (parent / "evaluate.py").is_file():
            return parent
    raise FileNotFoundError("Could not find assignment root containing data and evaluate.py")


def main() -> None:
    root = project_root()
    sys.path.insert(0, str(root))
    from starter import save_predictions

    medoid = np.load(
        Path(__file__).resolve().parent / "generic_medoid.npy",
        allow_pickle=False,
    )
    if medoid.dtype != np.float32 or medoid.shape != (2048, 3):
        raise ValueError(f"Invalid bundled generic medoid: {medoid.shape}, {medoid.dtype}")
    with np.load(root / "data" / "val.npz", allow_pickle=False) as val:
        sample_ids = val["sample_id"]
    predictions = np.repeat(np.asarray(medoid, dtype=np.float32)[None], len(sample_ids), axis=0)

    output_dir = root / "outputs" / "iteration4"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "generic_val_predictions.npz"
    save_predictions(prediction_path, sample_ids, predictions)
    metrics_path = Path(__file__).resolve().parents[1] / "metrics" / "val_metrics.json"
    command = [
        sys.executable,
        str(root / "evaluate.py"),
        str(prediction_path),
        "--save",
        str(metrics_path),
    ]
    print("Running official validation evaluator:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
