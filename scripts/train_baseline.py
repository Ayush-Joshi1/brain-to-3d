"""Create train-only generic and fMRI-conditioned validation predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics import medoid_shape
from starter import load_split, save_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "baseline")
    args = parser.parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    # Load only the permitted validation inputs; no validation targets enter fitting or prediction.
    with np.load(ROOT / "data" / "val.npz", allow_pickle=False) as archive:
        val_features = archive["X"]
        val_ids = archive["sample_id"]

    generic = medoid_shape(train["obj_points"])
    generic_predictions = np.repeat(generic[None, :, :], len(val_ids), axis=0)
    generic_path = args.output_dir / "generic_val_predictions.npz"
    save_predictions(generic_path, val_ids, generic_predictions)

    # Average time points, then standardize each voxel using training scans only.
    train_features = train["X"].mean(axis=1, dtype=np.float32)
    val_features = val_features.mean(axis=1, dtype=np.float32)
    mean = train_features.mean(axis=0, keepdims=True)
    scale = train_features.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train_features = (train_features - mean) / scale
    val_features = (val_features - mean) / scale

    distances = cdist(val_features, train_features, metric="sqeuclidean")
    nearest_train_scan = distances.argmin(axis=1)
    retrieved_groups = train["group"][nearest_train_scan]
    retrieval_predictions = train["obj_points"][retrieved_groups]
    retrieval_path = args.output_dir / "fmri_retrieval_val_predictions.npz"
    save_predictions(retrieval_path, val_ids, retrieval_predictions)

    run = {
        "seed": args.seed,
        "train_scans": int(len(train["sample_id"])),
        "train_objects": int(len(train["obj_points"])),
        "validation_scans": int(len(val_ids)),
        "validation_ground_truth_loaded": False,
        "feature": "mean over 10 time points; per-voxel z-score fitted on training scans only",
        "retrieval": "nearest standardized validation fMRI feature vector among training scans; output training scan's object point cloud",
        "generic": "official medoid_shape(train object point clouds)",
        "outputs": {
            "generic": str(generic_path.relative_to(ROOT)) if generic_path.is_relative_to(ROOT) else str(generic_path),
            "fmri_retrieval": str(retrieval_path.relative_to(ROOT)) if retrieval_path.is_relative_to(ROOT) else str(retrieval_path),
        },
    }
    run_path = args.output_dir / "baseline_run.json"
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
