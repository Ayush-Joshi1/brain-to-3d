"""Generate and verify the selected technical-fallback test predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter import save_predictions


def main() -> None:
    output_dir = ROOT / "outputs" / "iteration3"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / "outputs" / "baseline" / "generic_val_predictions.npz"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Expected saved train-only generic baseline: {baseline_path}")

    with np.load(baseline_path, allow_pickle=False) as baseline:
        baseline_ids = baseline["sample_id"]
        baseline_points = baseline["points"]
    if baseline_points.ndim != 3 or baseline_points.shape[1:] != (2048, 3):
        raise ValueError(f"Saved generic baseline has an unexpected point shape: {baseline_points.shape}")
    if not np.isfinite(baseline_points).all():
        raise ValueError("Saved generic baseline contains non-finite coordinates")
    if not len(baseline_points) or not all(
        np.array_equal(baseline_points[0], cloud) for cloud in baseline_points[1:]
    ):
        raise ValueError("Expected the saved generic baseline to repeat one medoid cloud")
    medoid = baseline_points[0].astype(np.float32, copy=True)

    with np.load(ROOT / "data" / "train.npz", allow_pickle=False) as train:
        train_shapes = train["obj_points"]
    exact_training_shape = any(np.array_equal(medoid, cloud) for cloud in train_shapes)
    if not exact_training_shape:
        raise ValueError("Saved generic cloud is not an exact member of the training object point-cloud table")

    test_path = ROOT / "data" / "test.npz"
    with np.load(test_path, allow_pickle=False) as test:
        test_ids = test["sample_id"].copy()
    if len(test_ids) != 167 or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError(f"Expected 167 unique test sample IDs, found {len(test_ids)}")
    if set(test_ids.tolist()) & set(baseline_ids.tolist()):
        raise ValueError("Test sample IDs unexpectedly overlap validation baseline IDs")

    predictions = np.repeat(medoid[None, :, :], len(test_ids), axis=0)
    output_path = output_dir / "test_predictions.npz"
    save_predictions(output_path, test_ids, predictions)

    with np.load(output_path, allow_pickle=False) as saved:
        keys = set(saved.files)
        saved_ids = saved["sample_id"]
        saved_points = saved["points"]
        if keys != {"sample_id", "points"}:
            raise ValueError(f"Unexpected prediction archive keys: {sorted(keys)}")
        if saved_points.shape != (167, 2048, 3) or saved_points.dtype != np.float32:
            raise ValueError(f"Invalid prediction array shape/dtype: {saved_points.shape}, {saved_points.dtype}")
        if not np.isfinite(saved_points).all():
            raise ValueError("Saved predictions contain NaN or infinite coordinates")
        if not np.array_equal(saved_ids, test_ids):
            raise ValueError("Saved sample IDs do not exactly match test archive ordering")
        if len(np.unique(saved_points.reshape(len(saved_points), -1), axis=0)) != 1:
            raise ValueError("Generic fallback should repeat one shape for every test scan")
        for index, cloud in enumerate(saved_points):
            if np.unique(cloud, axis=0).shape[0] < 100 or np.any(np.ptp(cloud, axis=0) <= 1e-6):
                raise ValueError(f"Malformed or empty point cloud at test index {index}")
        if not np.array_equal(saved_points[0], medoid):
            raise ValueError("Saved test cloud differs from the selected train-only medoid")

    manifest = {
        "prediction_archive": str(output_path.relative_to(ROOT)),
        "selection": "generic training medoid as a technical fallback; not a brain-conditioned solution",
        "source_validation_prediction": str(baseline_path.relative_to(ROOT)),
        "source_model_selection": "outputs/baseline/generic_metrics.json and Iteration 1 official validation report",
        "validation_metrics": {
            "chamfer": 0.15623311117483307,
            "fscore": 0.39467710665186084,
            "silhouette_iou": 0.43840225194248184,
            "silhouette_ssim": 0.6919060021340847,
            "identification": 48.763636363636365,
            "composite": -0.012759569677257923,
        },
        "uses_test_inputs": ["sample_id only"],
        "test_fmri_used": False,
        "archive_keys": ["sample_id", "points"],
        "sample_id_count": int(len(saved_ids)),
        "sample_ids_unique": True,
        "sample_ids_match_test_order_exactly": True,
        "points_shape": list(saved_points.shape),
        "points_dtype": str(saved_points.dtype),
        "points_finite": True,
        "nonempty_clouds": True,
        "unique_prediction_clouds": 1,
        "medoid_is_training_shape": True,
        "warning": "Predictions are intentionally identical across scans. The selected fallback is not evidence of fMRI-conditioned reconstruction and may fail the assignment brain-use gate.",
    }
    manifest_path = output_dir / "test_prediction_verification.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
