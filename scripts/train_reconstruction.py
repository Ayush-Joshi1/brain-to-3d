"""Fit a train-only fMRI-to-shape model and write validation predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter import save_predictions


def log(message: str, started: float) -> None:
    print(f"[{time.perf_counter() - started:7.1f}s] {message}", flush=True)


def memory_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024**2), 1)
    except ImportError:
        return None


def mean_features(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1, dtype=np.float32)


def make_occupancy(points: np.ndarray, lower: np.ndarray, upper: np.ndarray, resolution: int) -> np.ndarray:
    normalized = (points - lower) / (upper - lower)
    indices = np.floor(normalized * resolution).astype(np.int32)
    indices = np.clip(indices, 0, resolution - 1)
    occupied = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    return occupied.reshape(-1)


def decode_occupancies(
    predicted_codes: np.ndarray,
    shape_pca: PCA,
    lower: np.ndarray,
    upper: np.ndarray,
    resolution: int,
    seed: int,
) -> np.ndarray:
    scores = shape_pca.inverse_transform(predicted_codes)
    scores = np.clip(scores, 0.0, 1.0)
    points_per_scan = 2048
    axis_centers = lower[None, :] + (np.arange(resolution, dtype=np.float32)[:, None] + 0.5) * (
        (upper - lower)[None, :] / resolution
    )
    decoded: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for scan_scores in scores:
        weights = scan_scores.astype(np.float64)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Decoded occupancy has no finite positive mass")
        probabilities = weights / total
        selected = rng.choice(len(weights), size=points_per_scan, replace=True, p=probabilities)
        ijk = np.stack(np.unravel_index(selected, (resolution,) * 3), axis=1)
        cloud = axis_centers[ijk, np.arange(3)]
        jitter = rng.uniform(-0.5, 0.5, size=cloud.shape).astype(np.float32)
        cloud += jitter * ((upper - lower) / resolution)
        decoded.append(cloud.astype(np.float32))
    return np.stack(decoded)


def predict_validation(
    checkpoint: dict[str, object],
    x: np.ndarray,
    seed: int,
    decoder: str,
) -> np.ndarray:
    features = mean_features(x)
    scaler = checkpoint["feature_scaler"]
    fmri_pca = checkpoint["fmri_pca"]
    regressor = checkpoint["regressor"]
    predicted_codes = regressor.predict(fmri_pca.transform(scaler.transform(features)))
    if decoder == "codebook":
        nearest = cdist(predicted_codes, checkpoint["object_codes"], metric="sqeuclidean").argmin(axis=1)
        return np.asarray(checkpoint["object_points"][nearest], dtype=np.float32)
    return decode_occupancies(
        predicted_codes,
        checkpoint["shape_pca"],
        checkpoint["lower"],
        checkpoint["upper"],
        int(checkpoint["resolution"]),
        seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "iteration2")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fmri-components", type=int, default=64)
    parser.add_argument("--shape-components", type=int, default=24)
    parser.add_argument("--grid-resolution", type=int, default=16)
    parser.add_argument("--decoder", choices=("occupancy", "codebook"), default="occupancy")
    parser.add_argument("--prediction-name", default="validation_predictions.npz")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--predict-only", action="store_true", help="load the saved checkpoint and regenerate predictions")
    args = parser.parse_args()
    if args.grid_resolution < 4 or args.fmri_components < 1 or args.shape_components < 1:
        parser.error("component counts must be positive and grid resolution at least 4")

    started = time.perf_counter()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "reconstruction_model.joblib"
    prediction_path = args.output_dir / args.prediction_name
    config_path = args.output_dir / "config.json"

    if args.predict_only:
        checkpoint = joblib.load(checkpoint_path)
        with np.load(ROOT / "data" / "val.npz", allow_pickle=False) as val:
            val_x = val["X"]
            val_ids = val["sample_id"]
        predictions = predict_validation(checkpoint, val_x, int(checkpoint["seed"]), args.decoder)
        save_predictions(prediction_path, val_ids, predictions)
        log(f"Regenerated {len(val_ids)} predictions from checkpoint {checkpoint_path}", started)
        return

    log("Loading only training fMRI, groups, and object point clouds", started)
    with np.load(ROOT / "data" / "train.npz", allow_pickle=False) as train:
        train_x = train["X"]
        train_groups = train["group"]
        train_shapes = train["obj_points"]
    train_features = mean_features(train_x)
    del train_x

    lower = train_shapes.min(axis=(0, 1)).astype(np.float32)
    upper = train_shapes.max(axis=(0, 1)).astype(np.float32)
    margin = np.maximum((upper - lower) * 0.02, 1e-4)
    lower -= margin
    upper += margin
    resolution = args.grid_resolution
    object_occupancies = np.stack(
        [make_occupancy(cloud, lower, upper, resolution) for cloud in train_shapes]
    )
    log(
        f"Built {len(object_occupancies)} train-only {resolution}^3 occupancy codes "
        f"(RSS {memory_mb()} MiB)",
        started,
    )
    shape_components = min(args.shape_components, len(object_occupancies) - 1)
    shape_pca = PCA(n_components=shape_components, svd_solver="randomized", random_state=args.seed)
    object_codes = shape_pca.fit_transform(object_occupancies)
    scan_codes = object_codes[train_groups]
    log(
        f"Fit shape PCA ({shape_components} components; variance explained "
        f"{shape_pca.explained_variance_ratio_.sum():.3f})",
        started,
    )

    feature_scaler = StandardScaler(copy=False)
    scaled_features = feature_scaler.fit_transform(train_features)
    fmri_components = min(args.fmri_components, scaled_features.shape[0] - 1)
    fmri_pca = PCA(n_components=fmri_components, svd_solver="randomized", random_state=args.seed)
    reduced_features = fmri_pca.fit_transform(scaled_features).astype(np.float32, copy=False)
    log(
        f"Fit train-only fMRI standardization and PCA ({fmri_components} components; "
        f"variance explained {fmri_pca.explained_variance_ratio_.sum():.3f})",
        started,
    )

    unique_groups = np.unique(train_groups)
    folds = min(5, len(unique_groups))
    cv = GroupKFold(n_splits=folds)
    cv_results: list[dict[str, object]] = []
    for alpha in args.alphas:
        fold_losses: list[float] = []
        for fit_idx, held_idx in cv.split(reduced_features, scan_codes, groups=train_groups):
            model = Ridge(alpha=alpha)
            model.fit(reduced_features[fit_idx], scan_codes[fit_idx])
            residual = model.predict(reduced_features[held_idx]) - scan_codes[held_idx]
            fold_losses.append(float(np.mean(np.square(residual))))
        cv_results.append({"alpha": float(alpha), "fold_mse": fold_losses, "mean_mse": float(np.mean(fold_losses))})
        log(f"Grouped CV alpha={alpha:g}: latent MSE {np.mean(fold_losses):.6f}", started)
    best_alpha = min(cv_results, key=lambda item: float(item["mean_mse"]))["alpha"]
    regressor = Ridge(alpha=float(best_alpha))
    regressor.fit(reduced_features, scan_codes)
    checkpoint: dict[str, object] = {
        "seed": int(args.seed),
        "feature_scaler": feature_scaler,
        "fmri_pca": fmri_pca,
        "shape_pca": shape_pca,
        "regressor": regressor,
        "object_codes": object_codes,
        "object_points": train_shapes.astype(np.float32, copy=False),
        "lower": lower,
        "upper": upper,
        "resolution": resolution,
        "training_group_semantics": "split-local object table indices; used only to map training scans to training shapes and group CV",
        "feature_definition": "mean across 10 fMRI time points; scaler and randomized PCA fit on training scans only",
        "shape_definition": "PCA of binary 16^3 train-object occupancy grids; decoder emits weighted voxel-center samples",
        "selected_alpha": float(best_alpha),
        "cv_results": cv_results,
    }
    joblib.dump(checkpoint, checkpoint_path, compress=3)
    config = {
        "seed": args.seed,
        "fmri_components": fmri_components,
        "shape_components": shape_components,
        "grid_resolution": resolution,
        "decoder": args.decoder,
        "candidate_alphas": [float(alpha) for alpha in args.alphas],
        "selected_alpha": float(best_alpha),
        "grouped_cv_folds": folds,
        "grouped_cv_metric": "mean squared error in training-only shape PCA latent space",
        "shape_pca_variance_explained": float(shape_pca.explained_variance_ratio_.sum()),
        "fmri_pca_variance_explained": float(fmri_pca.explained_variance_ratio_.sum()),
        "training_scans": int(len(train_groups)),
        "training_objects": int(len(train_shapes)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "prediction_file": str(prediction_path.relative_to(ROOT)),
        "validation_targets_used_for_fit": False,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log(f"Saved checkpoint and configuration (RSS {memory_mb()} MiB)", started)
    del train_features, scaled_features, reduced_features, scan_codes, object_codes, object_occupancies

    with np.load(ROOT / "data" / "val.npz", allow_pickle=False) as val:
        val_x = val["X"]
        val_ids = val["sample_id"]
    predictions = predict_validation(checkpoint, val_x, args.seed, args.decoder)
    if predictions.shape != (len(val_ids), 2048, 3) or not np.isfinite(predictions).all():
        raise ValueError(f"Invalid validation predictions: {predictions.shape}, finite={np.isfinite(predictions).all()}")
    save_predictions(prediction_path, val_ids, predictions)
    log(f"Saved finite predictions {predictions.shape} to {prediction_path}", started)
    log("Training and prediction completed", started)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
