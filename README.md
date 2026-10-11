# Assignment: Reconstructing 3D objects from brain activity

## Background

Participants lay in an MRI scanner and watched short videos of single 3D objects rotating on a turntable. For each
video we have one fMRI recording of their brain activity. Your job is to **reconstruct the 3D shape of the object a
person was looking at, from the fMRI alone**, as a point cloud.

The data come from the public fMRI-Shape dataset (Gao et al., MinD-3D / MinD-3D++;
https://huggingface.co/datasets/Fudan-fMRI/fMRI-Shape). This subset has five categories: airplane, bench, cabinet,
car and sofa.

This is a hard, noisy problem: brain signals carry only coarse information about the object. Grading is fully
automatic, using the metrics below. A submission that ignores the fMRI is detected and scores zero.

## Data (`data/`)

| File | Contents |
|---|---|
| `train.npz` | 533 scans of 292 objects. Scans: `X`, `subject`, `group`, `sample_id`. Objects (indexed by `group`): `obj_points`, `obj_masks`, `obj_images`, `obj_class`, plus `class_names` |
| `val.npz` | 125 scans of 67 objects, same fields |
| `test.npz` | 167 scans: `X`, `subject`, `sample_id` only. **No shapes, images or labels.** |
| `cameras.npz` | the 32 calibrated stimulus cameras: `R`, `t`, `focal`, `cx`, `cy` (at 64×64), `video_frame` |
| `voxels.npz` | each voxel's position on a 256×256 flattened map of the cortex |

- **`X`**: float16, (scans, 10, 31141). 10 time points of the BOLD signal for 31,141 cortical voxels, with no
  normalisation applied. All subjects share the same voxel layout. Subjects differ systematically, and some appear
  only in val or test.
- **`obj_points`**: (2048, 3) ground-truth surface points. The ground truth is the object's **visual hull**: the 3D
  shape consistent with all 32 silhouettes. It is close to the true shape but fills in concavities.
- **`obj_masks`** (32, 64, 64) and **`obj_images`** (32, 64, 64, 3): silhouettes and frames of the stimulus video at the
  32 camera poses. Use them for training only; at test time you have fMRI only.
- **Coordinate frame:** every object is centred at the origin, z up, in the same canonical orientation and scale.
  Predictions must use this frame.
- **`group`**: an anonymous object id. **No object appears in more than one of train, val and test.** Group your
  cross-validation by it.

Total size about 460 MB. A good solution does not need a GPU.

## Submission

Submit one folder (zipped) with exactly this layout:

```
submission/
  predictions.npz                      sample_id (167,) and points (167, 2048, 3), float32, ground-truth frame
  ply/<sample_id>.ply                  167 point clouds, 2048 vertices each (same points as predictions.npz)
  renders/<sample_id>_view00.png       4 renders per scan, 256x256: views 00, 08, 16, 24
  renders/<sample_id>_view08.png       (stimulus cameras at 0, 90, 180 and 270 degrees)
  ...
  metrics/val_metrics.json             your validation metrics, written by evaluate.py
  code/                                runnable code
  report.pdf (or report.ipynb)
```

The helpers produce the files in the required format:

```python
from starter import save_predictions, export_outputs
save_predictions("submission/predictions.npz", test["sample_id"], test_points)
export_outputs("submission", test["sample_id"], test_points)      # writes ply/ and renders/
```
```bash
python evaluate.py val_predictions.npz --save submission/metrics/val_metrics.json
python check_submission.py submission/          # must print "Submission OK" before you submit
```

## Metrics (`metrics.py`, identical for validation and the hidden test set)

| Metric | Meaning | Better |
|---|---|---|
| **Chamfer distance** | mean distance prediction → true surface + true surface → prediction | lower |
| **F-score @ 0.05** | precision/recall of points within 0.05 of the other surface | higher |
| **Silhouette IoU** | your points drawn through the 32 cameras vs the true silhouettes | higher |
| **Silhouette SSIM** | structural similarity of the same silhouettes | higher |
| **Identification** | how often your prediction is closer to its own object than to the other test objects (chance 50%) | higher |

**Composite score** (ranking) = 0.5 × identification + 0.5 × mean of the four geometry metrics, each rescaled so
that **0 = the generic baseline** (one average training shape for every scan) and **1 = the best possible retrieval**
(for every scan, the training shape closest to its true shape).

**Brain-use gate:** if identification is not significantly above 50%, the submission is marked as generic output,
because the same shape is produced regardless of the scan. Note that the average shape already scores reasonably on
Chamfer, IoU and SSIM. The challenge is to beat it while making the output depend on the brain.

## What to do

1. Build any model that maps fMRI to a 2048-point cloud. Choose everything using train and validation (grouped by `group`).
2. Report validation metrics with uncertainty, and compare with the generic baseline (`evaluate.py` prints the composite).
3. Show that your output depends on the brain: for example, compare with shuffled fMRI or the average shape.
4. Predict the test set.

## Deliverables

Everything in the submission folder above:

- **3D outputs**: `predictions.npz` and one `.ply` point cloud per test scan.
- **2D renders**: four rendered views per test scan.
- **Metrics**: `metrics/val_metrics.json` (Chamfer, F-score, silhouette IoU, silhouette SSIM, identification, composite on
  validation). Test metrics are computed by us from your `predictions.npz` with the same code.
- **Code**, runnable end to end, with the expected runtime noted.
- **Report** (at most 3 pages): approach, validation metrics, controls, and what did and did not work.

## Rules

- At test time use only the test fMRI (and subject ids). Images, masks and points are for training.
- Use only the files provided; do not try to identify the objects or download the original 3D models.
- State any use of pretrained models.

## Licence and citation

A subset of fMRI-Shape, Apache-2.0 licence (`LICENSE_fMRI-Shape.txt`). Please cite Gao et al., *MinD-3D*
(arXiv 2312.07485) and *MinD-3D++* (arXiv 2409.11315).
