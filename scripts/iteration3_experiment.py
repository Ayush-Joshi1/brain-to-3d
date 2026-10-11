"""One grouped-CV diagnostic: global versus subject-centered fMRI features."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_reconstruction import make_occupancy, mean_features


def center_by_training_subject(
    fit_features: np.ndarray,
    fit_subjects: np.ndarray,
    apply_features: np.ndarray,
    apply_subjects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global_mean = fit_features.mean(axis=0, keepdims=True)
    subject_means = {
        subject: fit_features[fit_subjects == subject].mean(axis=0)
        for subject in np.unique(fit_subjects)
    }
    fit_centered = np.empty_like(fit_features)
    apply_centered = np.empty_like(apply_features)
    for subject in np.unique(fit_subjects):
        mask = fit_subjects == subject
        fit_centered[mask] = fit_features[mask] - subject_means[subject]
    for subject in np.unique(apply_subjects):
        mask = apply_subjects == subject
        baseline = subject_means.get(subject, global_mean[0])
        apply_centered[mask] = apply_features[mask] - baseline
    return fit_centered, apply_centered


def main() -> None:
    started = time.perf_counter()
    seed, resolution, shape_components, feature_components = 17, 16, 24, 32
    with np.load(ROOT / "data" / "train.npz", allow_pickle=False) as archive:
        x = mean_features(archive["X"])
        subjects = archive["subject"]
        groups = archive["group"]
        shapes = archive["obj_points"]

    results: dict[str, list[float]] = {
        "global": [],
        "subject_centered": [],
    }
    fold_details = []
    cv = GroupKFold(n_splits=5)
    for fold, (fit_indices, held_indices) in enumerate(
        cv.split(x, groups=groups), start=1
    ):
        fit_groups = np.unique(groups[fit_indices])
        held_groups = np.unique(groups[held_indices])
        lower = shapes[fit_groups].min(axis=(0, 1)).astype(np.float32)
        upper = shapes[fit_groups].max(axis=(0, 1)).astype(np.float32)
        margin = np.maximum((upper - lower) * 0.02, 1e-4)
        lower -= margin
        upper += margin
        fit_occupancies = np.stack(
            [make_occupancy(shapes[group], lower, upper, resolution) for group in fit_groups]
        )
        held_occupancies = np.stack(
            [make_occupancy(shapes[group], lower, upper, resolution) for group in held_groups]
        )
        shape_pca = PCA(
            n_components=shape_components,
            svd_solver="randomized",
            random_state=seed + fold,
        )
        fit_object_codes = shape_pca.fit_transform(fit_occupancies)
        fit_code_by_group = dict(zip(fit_groups, fit_object_codes, strict=True))
        fit_targets = np.stack([fit_code_by_group[group] for group in groups[fit_indices]])
        held_targets = shape_pca.transform(held_occupancies)
        held_code_by_group = dict(zip(held_groups, held_targets, strict=True))
        held_scan_targets = np.stack([held_code_by_group[group] for group in groups[held_indices]])

        fold_losses = {}
        for strategy in ("global", "subject_centered"):
            if strategy == "global":
                fit_raw, held_raw = x[fit_indices], x[held_indices]
            else:
                fit_raw, held_raw = center_by_training_subject(
                    x[fit_indices],
                    subjects[fit_indices],
                    x[held_indices],
                    subjects[held_indices],
                )
            scaler = StandardScaler()
            fit_scaled = scaler.fit_transform(fit_raw).astype(np.float32, copy=False)
            held_scaled = scaler.transform(held_raw).astype(np.float32, copy=False)
            feature_pca = PCA(
                n_components=feature_components,
                svd_solver="randomized",
                random_state=seed + fold,
            )
            fit_reduced = feature_pca.fit_transform(fit_scaled).astype(np.float32, copy=False)
            held_reduced = feature_pca.transform(held_scaled).astype(np.float32, copy=False)
            model = Ridge(alpha=1000.0)
            model.fit(fit_reduced, fit_targets)
            mse = float(np.mean(np.square(model.predict(held_reduced) - held_scan_targets)))
            results[strategy].append(mse)
            fold_losses[strategy] = mse
            del fit_scaled, held_scaled, fit_reduced, held_reduced, feature_pca, scaler, model
        fold_details.append(
            {
                "fold": fold,
                "fit_scans": len(fit_indices),
                "held_scans": len(held_indices),
                "fit_objects": len(fit_groups),
                "held_objects": len(held_groups),
                "latent_mse": fold_losses,
            }
        )
        print(
            f"fold {fold}/5: global={fold_losses['global']:.6f}, "
            f"subject-centered={fold_losses['subject_centered']:.6f}; "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    summary = {
        "seed": seed,
        "feature_components": feature_components,
        "shape_components": shape_components,
        "occupancy_resolution": resolution,
        "ridge_alpha": 1000.0,
        "folds": fold_details,
        "mean_latent_mse": {
            key: float(np.mean(value)) for key, value in results.items()
        },
        "subject_centered_relative_change": float(
            (np.mean(results["subject_centered"]) - np.mean(results["global"]))
            / np.mean(results["global"])
        ),
        "preprocessing_fitted_inside_each_grouped_fold": True,
        "shape_codebook_and_pca_fitted_inside_each_grouped_fold": True,
        "validation_targets_used": False,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "interpretation": "Exploratory training-only latent-MSE comparison; not official geometry evaluation and not a statistical test.",
    }
    out = ROOT / "outputs" / "iteration3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "subject_centering_experiment.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
