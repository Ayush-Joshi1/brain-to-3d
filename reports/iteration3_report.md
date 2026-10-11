# Iteration 3: targeted improvement and test predictions

## Decision and limitations

The best measured validation result remains the Iteration 1 generic training medoid. The Iteration 2 shape-codebook model is fMRI-conditioned in construction, but it scores below the generic baseline and its shuffled-association control scores higher. No candidate demonstrates credible scan-specific brain dependence. To deliver the required test archive, this iteration uses the generic medoid **only as an explicitly labeled technical fallback**. It is not presented as a successful solution to the brain-conditioned task and is likely to fail the assignment's brain-use gate.

No test ground-truth data exist in the supplied test archive, and no test metric is reported.

## Existing pipeline/control diagnostics

- `evaluate_iteration2.py` loads the prediction IDs, permutes only the rows of the prediction point clouds, and writes the original sample-ID vector unchanged. The generated control has exactly the same ordered 125 validation IDs as its source prediction file; only its shape-to-scan associations change.
- Both the unshuffled and shuffled files are passed through the same `evaluate.py` command. The evaluator aligns prediction rows to validation order by `sample_id` before computing metrics. Thus the observed difference is not caused by a changed ID set or a different metric path.
- The current checkpoint's 125 predicted shape-latent vectors are all distinct (total component variance 15.896; mean component variance 0.662). After nearest-codebook decoding, only 22 distinct point clouds remain. The output is variable, but the shuffled control shows that this alone does not demonstrate useful brain dependence.
- The saved Iteration 2 selected validation artifact was reproduced exactly from `reconstruction_model.joblib` and the validation fMRI. The Iteration 1 generic validation artifact contains exactly one repeated cloud and its sample IDs match the validation archive in order. The original training script uses the official `metrics.medoid_shape` utility. This iteration reused the saved training-derived medoid rather than repeating its relatively expensive medoid calculation.
- The occupancy decoder's generated predictions passed its `(125, 2048, 3)` / finite checks in Iteration 2, and the selected codebook output likewise passed. The train/test code uses positional order from each archive and does not sort IDs.
- A methodological issue was found in Iteration 2's grouped alpha selection: feature scaler/PCA and shape PCA were fitted once before the group folds, so held-out training-fold observations influenced unsupervised transforms. This did not use the separate validation split, but makes that internal CV estimate less strictly held out than intended. No official validation/test transform was fit on validation/test targets.
- The canonical point-cloud coordinate frame is preserved by reusing the official medoid cloud unchanged. The generic validation and test archives use the exact `sample_id` and `points` keys defined by `starter.save_predictions`.

## One focused experiment

Tested whether subtracting train-fold subject means helps the existing latent Ridge model. For each of five `GroupKFold` splits, all of the following were fit using only fold-training data: subject means, feature standardization, fMRI PCA, occupancy bounds, and shape PCA/codebook. Groups were the training split's local object-table indices. Held-out object scans remained entirely in the fold evaluation side. Used 32 fMRI components, 24 shape components, a 16³ occupancy grid, Ridge alpha 1000, and seed 17. This is a training-only latent-MSE diagnostic, not official geometry scoring.

| Group fold | Global feature preprocessing MSE | Subject-centered MSE |
|---:|---:|---:|
| 1 | 5.5031 | 5.4607 |
| 2 | 4.8712 | 4.8740 |
| 3 | 5.9402 | 5.9020 |
| 4 | 5.2325 | 5.3074 |
| 5 | 4.6987 | 4.7816 |
| Mean | 5.2491 | 5.2651 |

Subject-centering was about 0.3% worse on mean held-out latent MSE. It did not justify the added preprocessing or a new official validation evaluation; tuning stopped after this single focused attempt. This small CV comparison is not a statistical test.

## Official validation comparisons

Metrics below are measured by the supplied `evaluate.py` on 125 validation scans. No Iteration 3 candidate was newly evaluated because the sole focused training-only experiment did not improve its selection criterion.

| Candidate | Chamfer (lower) | F-score (higher) | Silhouette IoU (higher) | Silhouette SSIM (higher) | Identification | Composite |
|---|---:|---:|---:|---:|---:|---:|
| Iteration 1 generic training medoid | 0.1562 | 0.3947 | 0.4384 | 0.6919 | 48.76% | -0.013 |
| Iteration 1 standardized-fMRI nearest-scan retrieval | 0.2060 | 0.3253 | 0.3633 | 0.6417 | 49.48% | -0.158 |
| Iteration 2 occupancy decoder | 0.1622 | 0.4050 | 0.2590 | 0.2870 | 50.30% | -0.503 |
| Iteration 2 predicted-latent codebook | 0.2211 | 0.2659 | 0.2939 | 0.6844 | 52.59% | -0.141 |
| Iteration 2 codebook, one shuffled association | 0.2105 | 0.2874 | 0.3187 | 0.6918 | 57.59% | -0.055 |

The shuffled control preserves exact IDs and uses the same official evaluation. Its higher identification/composite is evidence against interpreting the unshuffled codebook's 52.59% identification as learned brain dependence. The control uses one permutation only; it does not provide a significance test or confidence interval.

## Test prediction artifact

Created `outputs/iteration3/test_predictions.npz` using the single train-derived medoid cloud repeated for every test ID. The generation script reads the saved generic validation baseline to recover the medoid, checks that it is an exact member of the training object point-cloud table, and reads only test sample IDs. It does not load test fMRI or read test labels, shapes, masks, or images because the selected technical fallback does not use test brain data.

Reload verification passed:

- Archive keys exactly `sample_id` and `points`.
- 167 IDs, unique and exactly equal to `data/test.npz` IDs in original order.
- `points` shape `(167, 2048, 3)`, dtype `float32`, all finite.
- Each cloud has more than 100 unique points and nonzero extent on every coordinate axis.
- All 167 clouds equal the verified training medoid; this is intentional generic fallback behavior.

The detailed machine-readable checks are in `outputs/iteration3/test_prediction_verification.json`. `check_submission.py` was not run: no complete submission tree with PLY, renders, validation metrics, code directory, and report PDF exists yet.

## Reproducibility and runtime

PowerShell commands from the workspace root:

```powershell
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\iteration3_experiment.py
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\train_reconstruction.py --predict-only --decoder codebook --prediction-name validation_predictions_best.npz
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\generate_test_predictions.py
```

The fold-aware subject experiment took 22.91 seconds. Iteration 2's saved-checkpoint reproduction took about 1.2 seconds. The final test artifact generation and verification took about 1.04 seconds. No dependencies were installed. Existing Iteration 1 and 2 outputs remain under their original directories.

## Iteration 4 remaining tasks

1. Decide whether to replace the generic fallback with a better, demonstrably brain-dependent model; current evidence does not validate one.
2. Generate test PLY files and four camera renders per test scan from the verified test predictions.
3. Assemble the required submission directory, include runnable code and the validation metric JSON, and create the assignment report PDF/Notebook.
4. Run `check_submission.py` against that complete staging directory and fix any reported layout or consistency problems before packaging.

Do not describe the current generic test output as fMRI reconstruction. It is a reproducible technical backup, not a model that passes the brain-use requirement.
