"""Reproducibly inspect the assignment archives without changing them."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_FILES = (
    "train.npz",
    "val.npz",
    "test.npz",
    "cameras.npz",
    "voxels.npz",
)
SPLIT_KEYS = {
    "train": {"X", "subject", "group", "sample_id", "obj_points", "obj_masks", "obj_images", "obj_class", "class_names"},
    "val": {"X", "subject", "group", "sample_id", "obj_points", "obj_masks", "obj_images", "obj_class", "class_names"},
    "test": {"X", "subject", "sample_id"},
}
EXPECTED_CLASSES = {"airplane", "bench", "cabinet", "car", "sofa"}


def _memory_gb() -> float | None:
    if os.name != "nt":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("avail_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.total_phys / (1024**3), 2)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _shape_fingerprint(points: np.ndarray) -> str:
    # Sorting coordinates makes this exact-geometry check independent of point order.
    ordered = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    return hashlib.sha256(np.ascontiguousarray(ordered).view(np.uint8)).hexdigest()


def audit() -> dict[str, object]:
    data_dir = ROOT / "data"
    missing = [str(data_dir / name) for name in EXPECTED_FILES if not (data_dir / name).is_file()]
    torch_version = _version("torch")
    nvidia_smi = shutil.which("nvidia-smi")
    result: dict[str, object] = {
        "root": str(ROOT),
        "expected_files_missing": missing,
        "archives": {},
        "checks": [],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_logical_count": os.cpu_count(),
            "ram_gb": _memory_gb(),
            "dependencies": {
                "numpy": _version("numpy"),
                "scipy": _version("scipy"),
                "Pillow": _version("Pillow"),
                "scikit-learn": _version("scikit-learn"),
                "torch": torch_version,
            },
            "cuda": (
                "CUDA runtime not available to this environment (PyTorch and nvidia-smi are absent)"
                if torch_version is None and nvidia_smi is None
                else "availability requires a CUDA-capable runtime check"
            ),
            "nvidia_smi": nvidia_smi,
        },
    }
    arrays_by_archive: dict[str, dict[str, np.ndarray]] = {}
    archive_bytes: dict[str, int] = {}
    for name in EXPECTED_FILES:
        path = data_dir / name
        if not path.is_file():
            continue
        arrays = dict(np.load(path, allow_pickle=False))
        arrays_by_archive[name[:-4]] = arrays
        archive_bytes[name[:-4]] = sum(a.nbytes for a in arrays.values())
        result["archives"][name] = {
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype), "nbytes": int(value.nbytes)}
                for key, value in arrays.items()
            },
            "uncompressed_total_bytes": archive_bytes[name[:-4]],
        }

    checks: list[dict[str, object]] = result["checks"]  # type: ignore[assignment]
    splits = {name: arrays_by_archive[name] for name in ("train", "val", "test") if name in arrays_by_archive}
    for split, arrays in splits.items():
        absent = sorted(SPLIT_KEYS[split] - arrays.keys())
        unexpected = sorted(arrays.keys() - SPLIT_KEYS[split])
        checks.append({"name": f"{split}_keys", "missing": absent, "unexpected": unexpected})
        expected_scans = {"train": 533, "val": 125, "test": 167}[split]
        expected_shapes = {
            "scan_count": len(arrays.get("sample_id", [])) == expected_scans,
            "fmri_shape": arrays.get("X", np.empty(0)).shape == (expected_scans, 10, 31141),
        }
        if split != "test":
            expected_shapes["point_cloud_shape"] = arrays.get("obj_points", np.empty(0)).shape[1:] == (2048, 3)
        checks.append({"name": f"{split}_expected_characteristics", "checks": expected_shapes})
        ids = arrays.get("sample_id")
        if ids is not None:
            unique_count = len(np.unique(ids))
            checks.append(
                {
                    "name": f"{split}_sample_id_uniqueness",
                    "count": len(ids),
                    "unique_count": unique_count,
                    "duplicates": int(len(ids) - unique_count),
                }
            )
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.number):
                checks.append(
                    {
                        "name": f"{split}_{key}_finite",
                        "nan_count": int(np.isnan(value).sum()),
                        "infinite_count": int(np.isinf(value).sum()),
                    }
                )
        if split in ("train", "val") and {"group", "obj_points", "obj_class"} <= arrays.keys():
            groups = arrays["group"]
            checks.append(
                {
                    "name": f"{split}_objects",
                    "scan_count": len(groups),
                    "object_count": len(arrays["obj_points"]),
                    "group_count": int(len(np.unique(groups))),
                    "group_min": int(groups.min()),
                    "group_max": int(groups.max()),
                    "group_out_of_bounds": int(((groups < 0) | (groups >= len(arrays["obj_points"]))).sum()),
                    "class_counts_by_object": dict(sorted(Counter(map(str, arrays["obj_class"])).items())),
                    "class_names": list(map(str, arrays.get("class_names", []))),
                    "classes_match_expected": set(map(str, arrays["obj_class"])) == EXPECTED_CLASSES,
                }
            )
        if "subject" in arrays:
            checks.append(
                {
                    "name": f"{split}_subject_distribution",
                    "counts": dict(sorted(Counter(map(str, arrays["subject"])).items())),
                }
            )

    id_overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left in splits and right in splits:
            a, b = set(map(str, splits[left]["sample_id"])), set(map(str, splits[right]["sample_id"]))
            id_overlaps[f"{left}_{right}"] = sorted(a & b)
    checks.append({"name": "sample_id_split_overlap", "overlap_ids": id_overlaps})

    if "train" in splits and "val" in splits:
        train, val = splits["train"], splits["val"]
        train_fingerprints = {_shape_fingerprint(p) for p in train["obj_points"]}
        val_fingerprints = {_shape_fingerprint(p) for p in val["obj_points"]}
        exact_shape_overlap = sorted(train_fingerprints & val_fingerprints)
        checks.append(
            {
                "name": "train_val_object_group_overlap",
                "train_objects": len(train["obj_points"]),
                "val_objects": len(val["obj_points"]),
                "numeric_group_id_overlap": sorted(set(map(int, train["group"])) & set(map(int, val["group"]))),
                "note": "group values index split-local object tables; numeric collisions do not identify the same object",
                "exact_point_cloud_geometry_overlap_count": len(exact_shape_overlap),
                "exact_point_cloud_geometry_overlap_sha256": exact_shape_overlap,
                "identity_verification": "Exact point-cloud equality checked; the archives expose no shared global object ID, so independently sampled identity cannot be fully established from group numbers alone.",
            }
        )

    if "cameras" in arrays_by_archive:
        cams = arrays_by_archive["cameras"]
        for key, value in cams.items():
            if np.issubdtype(value.dtype, np.number):
                checks.append(
                    {
                        "name": f"cameras_{key}_finite",
                        "nan_count": int(np.isnan(value).sum()),
                        "infinite_count": int(np.isinf(value).sum()),
                    }
                )
        checks.append(
            {
                "name": "camera_metadata",
                "shapes": {key: list(value.shape) for key, value in cams.items()},
                "dtypes": {key: str(value.dtype) for key, value in cams.items()},
            }
        )
    if "voxels" in arrays_by_archive:
        voxels = arrays_by_archive["voxels"]
        for key, value in voxels.items():
            if np.issubdtype(value.dtype, np.number):
                checks.append(
                    {
                        "name": f"voxels_{key}_finite",
                        "nan_count": int(np.isnan(value).sum()),
                        "infinite_count": int(np.isinf(value).sum()),
                    }
                )
        checks.append(
            {
                "name": "voxel_metadata",
                "shapes": {key: list(value.shape) for key, value in voxels.items()},
                "dtypes": {key: str(value.dtype) for key, value in voxels.items()},
                "voxel_count": len(voxels.get("row", [])),
                "mask_true_count": int(voxels["mask"].sum()) if "mask" in voxels else None,
            }
        )
    subjects_by_split = {
        split: Counter(map(str, arrays["subject"]))
        for split, arrays in splits.items()
        if "subject" in arrays
    }
    subject_union = sorted({subject for counts in subjects_by_split.values() for subject in counts})
    checks.append(
        {
            "name": "subject_split_comparison",
            "subjects": subject_union,
            "counts_by_split": {
                split: {subject: int(counts.get(subject, 0)) for subject in subject_union}
                for split, counts in subjects_by_split.items()
            },
            "subjects_only_in": {
                split: sorted(
                    subject
                    for subject in counts
                    if all(subject not in other for other_split, other in subjects_by_split.items() if other_split != split)
                )
                for split, counts in subjects_by_split.items()
            },
        }
    )

    feature_bytes = 4 * sum(len(splits[name]["X"]) * splits[name]["X"].shape[2] for name in ("train", "val") if name in splits)
    result["memory_estimates"] = {
        "sum_uncompressed_archive_arrays_bytes": int(sum(archive_bytes.values())),
        "train_val_features_mean_time_float32_bytes": int(feature_bytes),
        "baseline_estimated_working_set_bytes": int(sum(archive_bytes.get(s, 0) for s in ("train", "val")) + 3 * feature_bytes),
        "note": "Working-set estimate includes uncompressed train/validation arrays plus three float32 scan-by-voxel feature matrices; process overhead and temporary metric arrays are excluded.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional path to save the JSON audit")
    args = parser.parse_args()
    report = audit()
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved audit to {args.output}")
    return 1 if report["expected_files_missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
