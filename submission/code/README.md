# Reproducing the submitted technical fallback

This bundle contains the assignment's supplied data-independent prediction helper script and its selection settings in `config.json`. The exact training-derived generic medoid is stored in `generic_medoid.npy`. It intentionally uses the official generic training medoid because no tested fMRI-conditioned candidate beat that validation baseline. **It does not demonstrate brain-dependent reconstruction and may fail the assignment's brain-use gate.**

## Environment

Use Python 3.10+ and install the listed packages (NumPy, SciPy, Pillow):

```powershell
python -m pip install -r submission\code\requirements.txt
```

Place the supplied `data\` directory and the original assignment `starter.py` and `metrics.py` beside the `submission\` directory. The script locates that assignment root by searching parent directories; it does not require a fixed absolute path.

## Generate predictions, PLY, and renders

From the assignment workspace root:

```powershell
python submission\code\generate_submission.py
```

The script loads the bundled medoid selected from training object point clouds, reads only test sample IDs, writes `submission\predictions.npz`, and calls the original `starter.export_outputs`. That helper creates one ASCII PLY per scan and PNG renders for calibrated camera indices 0, 8, 16, and 24. When the verified Iteration 3 prediction archive is present, its predictions are reused instead.

The saved outputs were generated from the Iteration 3 verified prediction archive. Recomputing the official medoid from scratch is slow on a CPU; the bundled small medoid artifact avoids that work.

## Evaluate validation / check the completed folder

The original, unmodified official scripts are expected at the assignment workspace root. Recompute the selected generic medoid's official validation metrics with:

```powershell
python submission\code\evaluate_validation.py
python check_submission.py submission\
```

This script combines the bundled medoid with validation IDs, then invokes the supplied evaluator (which loads validation targets for scoring). `val_metrics.json` in this bundle records that completed official generic-medoid evaluation. Do not describe the generic fallback as an fMRI-conditioned model.
