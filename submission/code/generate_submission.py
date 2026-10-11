"""Build generic-medoid submission predictions and calibrated PLY/renders."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "train.npz").is_file() and (parent / "starter.py").is_file():
            return parent
    raise FileNotFoundError("Could not find assignment root containing data/train.npz and starter.py")


def main() -> None:
    root = project_root()
    sys.path.insert(0, str(root))

    from starter import export_outputs, save_predictions

    started = time.perf_counter()
    output = Path(__file__).resolve().parents[1]
    with np.load(root / "data" / "test.npz", allow_pickle=False) as test:
        test_ids = test["sample_id"].copy()
    verified_predictions = root / "outputs" / "iteration3" / "test_predictions.npz"
    if verified_predictions.is_file():
        with np.load(verified_predictions, allow_pickle=False) as source:
            if not np.array_equal(source["sample_id"], test_ids):
                raise ValueError("Verified Iteration 3 prediction IDs differ from test archive order")
            points = source["points"].astype(np.float32, copy=False)
        source_description = str(verified_predictions.relative_to(root))
    else:
        medoid = np.load(Path(__file__).resolve().parent / "generic_medoid.npy", allow_pickle=False)
        if medoid.dtype != np.float32 or medoid.shape != (2048, 3):
            raise ValueError(f"Invalid bundled generic medoid: {medoid.shape}, {medoid.dtype}")
        points = np.repeat(medoid[None], len(test_ids), axis=0)
        source_description = "bundled training-derived generic_medoid.npy"
    if points.shape != (len(test_ids), 2048, 3) or not np.isfinite(points).all():
        raise ValueError(f"Invalid prediction data: shape={points.shape}, finite={np.isfinite(points).all()}")
    save_predictions(output / "predictions.npz", test_ids, points)
    export_outputs(output, test_ids, points)
    manifest = {
        "selection": "generic training medoid technical fallback; not a brain-conditioned solution",
        "uses_training": ["train.obj_points"],
        "uses_test": ["test.sample_id only"],
        "prediction_source": source_description,
        "predictions": {
            "sample_count": int(len(test_ids)),
            "point_shape": list(points.shape),
            "dtype": str(points.dtype),
        },
        "output": str(output),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "warning": "Predictions are identical across test scans and may fail the brain-use gate.",
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
