"""Run the supplied official evaluator against both baseline predictions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "outputs" / "baseline"


def main() -> None:
    for name, prediction in (
        ("generic", "generic_val_predictions.npz"),
        ("fmri_retrieval", "fmri_retrieval_val_predictions.npz"),
    ):
        pred_path = BASELINE_DIR / prediction
        metrics_path = BASELINE_DIR / f"{name}_metrics.json"
        command = [
            sys.executable,
            str(ROOT / "evaluate.py"),
            str(pred_path),
            "--save",
            str(metrics_path),
        ]
        print(f"\nEvaluating {name}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
