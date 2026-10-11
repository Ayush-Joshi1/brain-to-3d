"""
Starter helpers for the fMRI -> 3D assignment: loading data. No model is included.

    from starter import load_split, load_cameras, targets_for
    train = load_split("train")
    pts, masks = targets_for(train)        # per-scan ground truth (expanded from the object table)
    save_predictions("submission/predictions.npz", test["sample_id"], my_points)
    export_outputs("submission", test["sample_id"], my_points)   # ply/ and renders/ for every test scan
"""

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def load_split(split):
    """train/val: X (scans,10,31141) float16, subject, sample_id, group, obj_points, obj_masks, obj_images,
    obj_class, class_names. test: X, subject, sample_id."""
    with np.load(DATA / f"{split}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_cameras():
    with np.load(DATA / "cameras.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_voxels():
    with np.load(DATA / "voxels.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def targets_for(split_data):
    """Ground-truth point cloud (2048,3) and silhouettes (32,64,64) for every scan of a train/val split."""
    g = split_data["group"]
    return split_data["obj_points"][g], split_data["obj_masks"][g]


def save_predictions(path, sample_id, points):
    """points: (n_scans, 2048, 3) in the ground-truth frame, in the same order as sample_id."""
    np.savez_compressed(path, sample_id=np.asarray(sample_id), points=np.asarray(points, dtype=np.float32))


# ---------------------------------------------------------------- required output files
RENDER_VIEWS = (0, 8, 16, 24)       # camera indices (stimulus video frames 0, 48, 96, 144)
RENDER_RES = 256


def write_ply(path, points):
    """ASCII PLY with one vertex per point (x, y, z)."""
    points = np.asarray(points, dtype=np.float32)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, points, fmt="%.5f")


def read_ply(path):
    with open(path) as f:
        n = next(int(line.split()[-1]) for line in f if line.startswith("element vertex"))
        for line in f:
            if line.strip() == "end_header":
                break
        return np.loadtxt(f, dtype=np.float32, max_rows=n).reshape(n, 3)


def render_points(points, cams, view, res=RENDER_RES, radius=2):
    """Render a point cloud through one stimulus camera: grey discs on white, darker = closer."""
    from metrics import project
    scale = res / int(cams["resolution"])
    c = {"R": cams["R"][view:view + 1], "t": cams["t"][view:view + 1], "focal": cams["focal"] * scale,
         "cx": cams["cx"] * scale, "cy": cams["cy"] * scale}
    uv, depth = project(np.asarray(points, dtype=np.float32), c)
    uv, depth = uv[0], depth[0]
    img = np.full((res, res), 255, np.uint8)
    ok = depth > 0.05
    if ok.sum() == 0:
        return img
    d = depth[ok]
    shade = (40 + 150 * (d - d.min()) / max(d.max() - d.min(), 1e-6)).astype(np.uint8)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx ** 2 + yy ** 2) <= radius ** 2
    x = (np.floor(uv[ok, 0]).astype(int)[:, None] + xx[disk][None]).ravel()
    y = (np.floor(uv[ok, 1]).astype(int)[:, None] + yy[disk][None]).ravel()
    s = np.repeat(shade, disk.sum())
    inside = (x >= 0) & (x < res) & (y >= 0) & (y < res)
    np.minimum.at(img, (y[inside], x[inside]), s[inside])                       # nearest point wins each pixel
    return img


def export_outputs(out_dir, sample_id, points):
    """Write the required files for every scan: ply/<id>.ply and renders/<id>_view<k>.png for k in RENDER_VIEWS."""
    from PIL import Image
    out = Path(out_dir)
    (out / "ply").mkdir(parents=True, exist_ok=True)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    cams = load_cameras()
    for sid, pts in zip(sample_id, points):
        write_ply(out / "ply" / f"{sid}.ply", pts)
        for v in RENDER_VIEWS:
            Image.fromarray(render_points(pts, cams, v)).save(out / "renders" / f"{sid}_view{v:02d}.png")


if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        d = load_split(s)
        msg = f"{s:<5} X {d['X'].shape}, subjects {len(set(d['subject']))}"
        if "group" in d:
            msg += f", objects {len(d['obj_points'])}, points {d['obj_points'].shape[1:]}, masks {d['obj_masks'].shape[1:]}"
        print(msg)
