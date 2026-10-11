"""Evaluate Iteration 2 predictions and a shuffled scan-to-prediction control."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "iteration2"


def main() -> None:
    with np.load(OUT / "validation_predictions.npz", allow_pickle=False) as archive:
        ids = archive["sample_id"]
        occupancy_points = archive["points"]
    rng = np.random.default_rng(17)
    codebook_path = OUT / "validation_predictions_codebook.npz"
    if codebook_path.exists():
        with np.load(codebook_path, allow_pickle=False) as archive:
            selected_decoder = "codebook"
            control_points = archive["points"]
    else:
        selected_decoder = "occupancy"
        control_points = occupancy_points
    permutation = rng.permutation(len(ids))
    shuffled_name = f"validation_predictions_{selected_decoder}_shuffled.npz"
    np.savez_compressed(OUT / shuffled_name, sample_id=ids, points=control_points[permutation])
    displacement = np.linalg.norm(
        control_points - control_points.mean(axis=0, keepdims=True), axis=2
    ).mean(axis=1)
    unique_clouds = len({cloud.tobytes() for cloud in control_points})
    control = {
        "seed": 17,
        "decoder": selected_decoder,
        "unique_prediction_clouds": unique_clouds,
        "validation_scan_count": len(control_points),
        "mean_per_scan_pointwise_distance_from_mean_cloud": float(displacement.mean()),
        "mean_pointwise_distance_std_across_scans": float(control_points.std(axis=0).mean()),
        "shuffled_prediction_indices": permutation.tolist(),
        "interpretation": "The shuffled evaluator pairs each target with a different validation scan's prediction; compare identification and geometry with the original evaluation.",
    }
    (OUT / "brain_dependence_control.json").write_text(json.dumps(control, indent=2) + "\n", encoding="utf-8")

    evaluations = [
        ("model", "validation_predictions.npz"),
        (f"{selected_decoder}_shuffled_control", shuffled_name),
    ]
    if selected_decoder == "codebook":
        evaluations.append(("codebook_retrieval", codebook_path.name))
    timings = {}
    for label, filename in evaluations:
        metrics_path = OUT / f"{label}_metrics.json"
        command = [
            sys.executable,
            str(ROOT / "evaluate.py"),
            str(OUT / filename),
            "--save",
            str(metrics_path),
        ]
        print(f"\nEvaluating {label}: {' '.join(command)}", flush=True)
        evaluation_started = time.perf_counter()
        subprocess.run(command, cwd=ROOT, check=True)
        timings[label] = round(time.perf_counter() - evaluation_started, 2)

    (OUT / "evaluation_runtime.json").write_text(
        json.dumps({"seconds_by_evaluation": timings, "total_seconds": round(sum(timings.values()), 2)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    candidates = {
        "occupancy": json.loads((OUT / "model_metrics.json").read_text(encoding="utf-8")),
    }
    if codebook_path.exists():
        candidates["codebook"] = json.loads(
            (OUT / "codebook_retrieval_metrics.json").read_text(encoding="utf-8")
        )
    chosen = max(candidates, key=lambda decoder: candidates[decoder]["composite"])
    selected_prediction = "validation_predictions_best.npz"
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["selected_validation_decoder"] = chosen
    config["selected_validation_composite"] = candidates[chosen]["composite"]
    config["selected_prediction"] = selected_prediction
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    selection = {
        "selection_metric": "official validation composite",
        "selected_decoder": chosen,
        "selected_prediction": selected_prediction,
        "checkpoint": "reconstruction_model.joblib",
        "validation_metrics": candidates[chosen],
        "candidate_composites": {
            decoder: metrics["composite"] for decoder, metrics in candidates.items()
        },
        "note": "Validation labels were used only by the official evaluator to select a decoder; no model or preprocessing parameters were fitted to validation targets.",
    }
    (OUT / "model_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nScan-specific prediction control:")
    print(json.dumps(control, indent=2))
    print(f"\nOfficial evaluation runtimes: {timings}")
    print("\nSelected Iteration 2 decoder:")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
