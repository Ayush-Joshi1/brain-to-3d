"""
Evaluation metrics for the fMRI -> 3D reconstruction assignment. Only numpy and scipy are needed.

A prediction is a point cloud of N_POINTS 3D points per scan, in the same coordinate frame as the provided
ground-truth `obj_points` (objects centred at the origin, z up, ShapeNet scale).

  chamfer          mean distance from predicted points to the nearest true point + mean distance from true points
                   to the nearest predicted point (lower is better)
  fscore           harmonic mean of precision and recall at distance TAU (higher is better)
  silhouette_iou   points are projected through the 32 calibrated stimulus cameras, drawn as small discs at 64x64,
                   and compared with the true silhouettes (higher is better)
  silhouette_ssim  structural similarity (SSIM) between the drawn and the true silhouettes, same 32 views (higher is better)
  identification   for each scan, the share of OTHER evaluated objects whose true shape is farther (by Chamfer) from the
                   prediction than the scan's own object. 50% = the prediction is not specific to the object.

  composite        0.5 * identification_norm + 0.5 * mean(chamfer_norm, fscore_norm, iou_norm, ssim_norm), where each
                   metric is rescaled so that 0 = the generic baseline (one average training shape for every scan) and
                   1 = the best possible retrieval (for every scan, the training shape closest to its true shape);
                   identification_norm uses 50% -> 0 and the retrieval ceiling -> 1. Ignoring the fMRI scores 0.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

N_POINTS = 2048
TAU = 0.05
SPLAT_RADIUS = 1.5          # pixels at 64x64


def chamfer_and_fscore(pred, gt, tau=TAU):
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    precision, recall = (d_pg < tau).mean(), (d_gp < tau).mean()
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return d_pg.mean() + d_gp.mean(), f


def project(points, cams):
    """points (N,3) -> pixel coordinates (V,N,2) and depth (V,N) for all cameras (camera looks down -z)."""
    pc = np.einsum("vij,nj->vni", cams["R"], points) + cams["t"][:, None, :]
    depth = -pc[..., 2]
    u = cams["cx"] + cams["focal"] * pc[..., 0] / depth
    v = cams["cy"] - cams["focal"] * pc[..., 1] / depth
    return np.stack([u, v], -1), depth


_DISK = None


def render_silhouettes(points, cams):
    """(V,64,64) float silhouettes: each point drawn as a disc of SPLAT_RADIUS pixels."""
    global _DISK
    res = int(cams["resolution"])
    if _DISK is None:
        r = int(np.ceil(SPLAT_RADIUS))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        _DISK = (xx ** 2 + yy ** 2) <= SPLAT_RADIUS ** 2
    uv, depth = project(points, cams)
    out = np.zeros((len(uv), res, res), bool)
    for k in range(len(uv)):
        ok = depth[k] > 0.05
        px = np.floor(uv[k][ok]).astype(int)
        inside = (px[:, 0] >= 0) & (px[:, 0] < res) & (px[:, 1] >= 0) & (px[:, 1] < res)
        out[k, px[inside, 1], px[inside, 0]] = True
        out[k] = ndimage.binary_dilation(out[k], structure=_DISK)
    return out.astype(np.float32)


def ssim(a, b, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    """Single-scale SSIM of two images in [0,1] with a Gaussian window."""
    f = lambda x: ndimage.gaussian_filter(x, sigma)
    mu_a, mu_b = f(a), f(b)
    var_a, var_b, cov = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def silhouette_scores(points, true_masks, cams):
    """true_masks (32,64,64) uint8 0-255 or float 0-1."""
    gt = true_masks.astype(np.float32)
    gt = gt / 255.0 if gt.max() > 1 else gt
    pr = render_silhouettes(points, cams)
    p, t = pr > 0.5, gt > 0.5
    iou = np.mean([(p[k] & t[k]).sum() / max((p[k] | t[k]).sum(), 1) for k in range(len(p))])
    s = np.mean([ssim(pr[k], gt[k]) for k in range(len(p))])
    return float(iou), s


def check_predictions(points, n_expected):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (n_expected, N_POINTS, 3):
        raise ValueError(f"points must have shape ({n_expected}, {N_POINTS}, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points contain NaN or infinite values")
    return points


def evaluate(pred_points, true_points, true_masks, object_index, cams):
    """
    pred_points (n,2048,3), true_points (n,M,3), true_masks (n,32,64,64), object_index (n,) identifies which scans
    share an object (used for identification). Returns (summary dict, per-scan list of dicts).
    """
    n = len(pred_points)
    per = []
    for k in range(n):
        cd, fs = chamfer_and_fscore(pred_points[k], true_points[k])
        iou, s = silhouette_scores(pred_points[k], true_masks[k], cams)
        per.append({"chamfer": cd, "fscore": fs, "silhouette_iou": iou, "silhouette_ssim": s})

    objs = np.unique(object_index)
    rep = {o: int(np.where(object_index == o)[0][0]) for o in objs}
    trees = {o: cKDTree(true_points[rep[o]]) for o in objs}
    for k in range(n):
        pred_tree = cKDTree(pred_points[k])
        cd = {o: trees[o].query(pred_points[k])[0].mean() + pred_tree.query(true_points[rep[o]])[0].mean() for o in objs}
        own = cd[object_index[k]]
        others = [cd[o] for o in objs if o != object_index[k]]
        per[k]["identification"] = 100.0 * np.mean([own < c for c in others]) if others else np.nan

    summary = {m: float(np.nanmean([p[m] for p in per])) for m in per[0]}
    return summary, per


def format_summary(s):
    return (f"Chamfer {s['chamfer']:.4f} (lower is better) | F-score@{TAU} {s['fscore']:.3f} | "
            f"silhouette IoU {s['silhouette_iou']:.3f} | silhouette SSIM {s['silhouette_ssim']:.3f} | "
            f"identification {s['identification']:.1f}% (chance 50%)")


GEOMETRY = ["chamfer", "fscore", "silhouette_iou", "silhouette_ssim"]


def medoid_shape(lib_points):
    """The training shape with the smallest mean Chamfer to all others (the generic baseline)."""
    trees = [cKDTree(p) for p in lib_points]
    D = np.array([[trees[j].query(lib_points[i])[0].mean() + trees[i].query(lib_points[j])[0].mean()
                   for j in range(len(lib_points))] for i in range(len(lib_points))])
    return lib_points[int(D.mean(1).argmin())]


def oracle_retrieval(lib_points, true_points):
    """For each target, the training shape closest to it (Chamfer)."""
    trees = [cKDTree(p) for p in lib_points]
    out = []
    for tp in true_points:
        tt = cKDTree(tp)
        out.append(lib_points[int(np.argmin([tr.query(tp)[0].mean() + tt.query(lp)[0].mean()
                                             for lp, tr in zip(lib_points, trees)]))])
    return np.stack(out)


def composite(summary, generic, oracle):
    norm = lambda m: (summary[m] - generic[m]) / (oracle[m] - generic[m])
    ident = (summary["identification"] - 50.0) / (oracle["identification"] - 50.0)
    return 0.5 * ident + 0.5 * float(np.mean([norm(m) for m in GEOMETRY]))
