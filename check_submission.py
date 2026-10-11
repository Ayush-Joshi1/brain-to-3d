"""
Check that a submission folder is complete and consistent (applicants: run this before submitting).

    python check_submission.py submission/

Expected layout:
  submission/predictions.npz            sample_id (167,), points (167, 2048, 3)
  submission/ply/<sample_id>.ply        167 files, 2048 vertices each, identical to predictions.npz
  submission/renders/<sample_id>_view{00,08,16,24}.png   668 images, 256x256
  submission/metrics/val_metrics.json   written by: python evaluate.py val_predictions.npz --save submission/metrics/val_metrics.json
  submission/code/, submission/report.pdf (or report.ipynb)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from metrics import N_POINTS
from starter import RENDER_RES, RENDER_VIEWS, load_split, read_ply

root = Path(sys.argv[1])
problems = []
test_ids = list(load_split("test")["sample_id"])

pred = None
if not (root / "predictions.npz").exists():
    problems.append("missing predictions.npz")
else:
    with np.load(root / "predictions.npz", allow_pickle=False) as z:
        ids, pts = list(z["sample_id"]), z["points"]
    if sorted(ids) != sorted(test_ids):
        problems.append("predictions.npz must contain exactly the 167 test sample_ids")
    elif pts.shape != (len(test_ids), N_POINTS, 3) or not np.isfinite(pts).all():
        problems.append(f"predictions.npz points must be finite with shape ({len(test_ids)}, {N_POINTS}, 3), got {pts.shape}")
    else:
        pred = dict(zip(ids, pts))

bad_ply, mismatch, bad_png = [], [], []
for sid in test_ids:
    f = root / "ply" / f"{sid}.ply"
    if not f.exists():
        bad_ply.append(sid)
        continue
    try:
        p = read_ply(f)
        if p.shape != (N_POINTS, 3):
            bad_ply.append(sid)
        elif pred is not None and np.abs(p - pred[sid]).max() > 1e-3:
            mismatch.append(sid)
    except Exception:
        bad_ply.append(sid)
    for v in RENDER_VIEWS:
        g = root / "renders" / f"{sid}_view{v:02d}.png"
        if not g.exists() or Image.open(g).size != (RENDER_RES, RENDER_RES):
            bad_png.append(g.name)
if bad_ply:
    problems.append(f"{len(bad_ply)} PLY files missing or not {N_POINTS} vertices (e.g. {bad_ply[:3]})")
if mismatch:
    problems.append(f"{len(mismatch)} PLY files differ from predictions.npz (e.g. {mismatch[:3]})")
if bad_png:
    problems.append(f"{len(bad_png)} renders missing or not {RENDER_RES}x{RENDER_RES} (e.g. {bad_png[:3]})")

m = root / "metrics" / "val_metrics.json"
if not m.exists():
    problems.append("missing metrics/val_metrics.json (run evaluate.py ... --save)")
else:
    keys = {"chamfer", "fscore", "silhouette_iou", "silhouette_ssim", "identification", "composite"}
    missing = keys - set(json.loads(m.read_text()))
    if missing:
        problems.append(f"val_metrics.json lacks {sorted(missing)}")
if not (root / "code").is_dir():
    problems.append("missing code/ folder")
if not any((root / n).exists() for n in ("report.pdf", "report.ipynb")):
    problems.append("missing report.pdf or report.ipynb")

if problems:
    print("SUBMISSION INCOMPLETE:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print(f"Submission OK: predictions.npz, {len(test_ids)} PLY files, {len(test_ids) * len(RENDER_VIEWS)} renders, "
      f"validation metrics, code and report.")
