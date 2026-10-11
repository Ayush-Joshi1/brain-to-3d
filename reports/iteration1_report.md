# Iteration 1: Dataset Audit, Environment Setup & Baseline

## Environment

- Windows 10, system Python 3.13.7 (`C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe`).
- Intel Core i7-6600U, 2 physical cores / 4 logical processors; 7.88 GiB installed RAM.
- Detected display adapter: Intel HD Graphics 520. No PyTorch or `nvidia-smi` is available, so CUDA is unavailable to this environment.
- Existing dependencies: NumPy 2.2.5, SciPy 1.17.1, Pillow 11.3.0, scikit-learn 1.9.0. No packages were installed. `requirements.txt` lists only the runtime dependencies used by this pipeline.

## Dataset inventory

All expected files are present under `data\`; the extra supplied `val_anchors.npz` is retained and used only by the official evaluator.

| Archive | Scans / array shapes | Compressed size | Uncompressed arrays |
|---|---|---:|---:|
| train.npz | `X` (533, 10, 31141); 292 object clouds (2048, 3), masks (292, 32, 64, 64), images (292, 32, 64, 64, 3) | 299.8 MB | 469.5 MiB |
| val.npz | `X` (125, 10, 31141); 67 object clouds (2048, 3), masks (67, 32, 64, 64), images (67, 32, 64, 64, 3) | 69.9 MB | 109.3 MiB |
| test.npz | `X` (167, 10, 31141), subject and sample IDs only | 86.3 MB | 99.2 MiB |
| cameras.npz | 32 rotations/translations, scalar intrinsics and 32 video-frame indices | 2.3 KB | 2.2 KB |
| voxels.npz | row/col (31141,), cortex mask (256, 256), 31141 mask voxels | 4.3 KB | 0.2 MB |

Training and validation object classes and clouds/masks/images are present. Object counts by class are train: airplane 58, bench 59, cabinet 60, car 58, sofa 57; validation: 13, 14, 14, 13, 13 respectively. Test has no labels, points, masks, or images. Expected scan counts, fMRI dimensions, and point-cloud dimensions all passed.

## Integrity and leakage checks

- All split sample IDs are unique; no sample IDs overlap between any pair of splits.
- All numeric split, camera, and voxel arrays have zero NaN and infinite values. Train and validation groups are in bounds and cover all their split-local object tables.
- The `group` numbers are indices into each split's own object table, not global object identifiers. Their numerical overlap (0–66) is therefore not evidence of object leakage. No exact train/validation point-cloud geometry duplicates were found by order-independent coordinate fingerprint. The archives provide no shared global object ID with which to independently verify identity beyond this check; the supplied assignment documentation states object groups do not cross splits.
- Subjects occur in all splits except `sub-0013`, which occurs in train only. Train/validation/test subject count distributions are recorded in `outputs\audit\dataset_audit.json`.
- Estimated baseline working set is about 813 MiB (uncompressed train/validation arrays plus standardized feature matrices); this excludes process overhead and evaluation temporaries.

## Dependencies and reproducibility

The active Python environment already had NumPy, SciPy, and Pillow, so no installation was necessary. If setting up another compatible environment, run:

```powershell
python -m pip install -r requirements.txt
python scripts\audit_dataset.py --output outputs\audit\dataset_audit.json
python scripts\train_baseline.py
python scripts\evaluate_baseline.py
```

The exact commands used for the audit, prediction generation, and official evaluations were:

```powershell
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\audit_dataset.py --output outputs\audit\dataset_audit.json
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\train_baseline.py
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\evaluate_baseline.py
```

The baseline sets NumPy's random seed to 0. It writes predictions and metrics under ignored `outputs\baseline\`; it does not modify the original data archives or the supplied grading scripts. Dependency versions were checked in the current environment; no install command was run.

## Baselines

1. **Generic:** the official `metrics.medoid_shape` representative from training object clouds, repeated for every validation scan.
2. **fMRI retrieval:** mean-pool the 10 fMRI time points; fit per-voxel means and standard deviations on training scans only; retrieve the closest standardized training scan and emit that scan's training object cloud. Validation input loading is restricted to `X` and `sample_id`. The training and validation object tables have no exact point-cloud duplicates.

Both prediction files use the official `sample_id` / float32 `points` format, with 2048 points in the supplied canonical coordinate frame. No test data are used by this validation pipeline.

### Official validation results

Both baselines completed through the unmodified `evaluate.py` on all 125 validation scans:

| Baseline | Chamfer ↓ | F-score ↑ | Silhouette IoU ↑ | Silhouette SSIM ↑ | Identification | Composite |
|---|---:|---:|---:|---:|---:|---:|
| Generic training medoid | 0.1562 | 0.3947 | 0.4384 | 0.6919 | 48.76% | -0.013 |
| Standardized-fMRI retrieval | 0.2060 | 0.3253 | 0.3633 | 0.6417 | 49.48% | -0.158 |

The fMRI retrieval baseline was worse than the generic medoid on all four geometry metrics and did not exceed chance identification. It is not a useful model yet and does not demonstrate reliable brain-conditioned reconstruction. These are point estimates; no uncertainty interval or shuffled-fMRI control was computed in this iteration. The generic composite is slightly below zero because its measured identification (48.76%) is below the composite's 50% identification reference.

## Runtime observations and unresolved issues

The training command completed, but the exact wall-clock duration was not instrumented. Selecting the exact training-cloud medoid was CPU-heavy on this two-core machine and remained active for over ten minutes; prediction and evaluation then completed. Running the official evaluator for both baselines took about 2.5 minutes. The approximate train/validation working set (813 MiB) fit in the available RAM.

The group arrays are split-local indices, and there is no shared global object ID; exact point-cloud fingerprints found no train/validation duplicates, but cannot rule out independently sampled copies of the same underlying object. No CUDA-capable runtime was detected. The fMRI retrieval metric is a deliberately simple baseline, not a tuned model.

## Iteration 2 recommendation

With only 533 training scans and 292 unique training shapes, begin with a compact, strongly regularized mapping rather than a large end-to-end network. Fit any fMRI normalization/PCA on train only, retain temporal information with a small temporal projection or 1-D temporal block, and include a subject embedding or train-only subject calibration to address subject shifts. A practical shape head is a PointNet-style encoder/decoder trained only on the training shapes, with a small ridge/MLP fMRI-to-shape-latent mapper. Optimize a combination of Chamfer and silhouette loss in the canonical frame. Validate by object group, compare against both baselines, and include shuffled-fMRI and generic-shape controls before scaling the model.
