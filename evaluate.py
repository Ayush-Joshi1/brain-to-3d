"""
Score predictions on the VALIDATION split (the organisers run the same metrics on the hidden test split).

    python evaluate.py val_predictions.npz [--save submission/metrics/val_metrics.json]

val_predictions.npz: sample_id (n,), points (n, 2048, 3), one entry per validation scan, in any order.
"""

import sys

import numpy as np

from pathlib import Path

from metrics import check_predictions, composite, evaluate, format_summary, medoid_shape, oracle_retrieval
from starter import load_cameras, load_split, targets_for

val = load_split("val")
with np.load(sys.argv[1], allow_pickle=False) as z:
    order = {s: k for k, s in enumerate(z["sample_id"])}
    if set(order) != set(val["sample_id"]):
        sys.exit(f"Predictions must cover exactly the {len(val['sample_id'])} validation scans")
    points = check_predictions(z["points"][[order[s] for s in val["sample_id"]]], len(val["sample_id"]))
true_points, true_masks = targets_for(val)
cams = load_cameras()
summary, _ = evaluate(points, true_points, true_masks, val["group"], cams)
print(format_summary(summary))

# reference anchors on validation (computed once, cached next to this script)
cache = Path(__file__).resolve().parent / "data" / "val_anchors.npz"
if not cache.exists():
    lib = load_split("train")["obj_points"]
    generic, _ = evaluate(np.repeat(medoid_shape(lib)[None], len(points), 0), true_points, true_masks, val["group"], cams)
    oracle, _ = evaluate(oracle_retrieval(lib, true_points), true_points, true_masks, val["group"], cams)
    np.savez(cache, **{f"generic_{k}": v for k, v in generic.items()}, **{f"oracle_{k}": v for k, v in oracle.items()})
with np.load(cache) as z:
    generic = {k[8:]: float(z[k]) for k in z.files if k.startswith("generic_")}
    oracle = {k[7:]: float(z[k]) for k in z.files if k.startswith("oracle_")}
score = composite(summary, generic, oracle)
print(f"Composite score {score:+.3f}  (0 = generic average shape, 1 = best possible retrieval)")
if "--save" in sys.argv:
    import json
    out = Path(sys.argv[sys.argv.index("--save") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**summary, "composite": score, "split": "val", "n_scans": len(points)}, indent=2))
    print(f"Saved {out}")
