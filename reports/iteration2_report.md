# Iteration 2: fMRI-to-3D reconstruction

## Result

The model pipeline fit and generated predictions successfully on CPU. The Iteration 2 decoder with the best validation composite was nearest-codebook retrieval in predicted shape-PCA space. It did **not** outperform the Iteration 1 generic medoid baseline. The shuffled association control scored better than the unshuffled selected model, so the available results do not provide evidence that its scan-to-shape mapping learned useful brain dependence.

## Model and rationale

The pipeline is a small scikit-learn regression model rather than a neural network:

1. Average each scan across the 10 time points, leaving a 31,141-dimensional fMRI vector.
2. Fit a per-feature `StandardScaler` and randomized 64-component PCA on training scans only. The PCA retained 95.4% of training fMRI variance.
3. Convert each unique training-object point cloud to a binary 16x16x16 occupancy grid in bounds computed from training shapes only. Fit a randomized 24-component shape PCA on the 292 training objects; these components explain 69.3% of occupancy variance.
4. Associate scans with object shape codes using the train split's local `group` table indices. Select Ridge alpha from `{0.1, 1, 10, 100, 1000}` using five-fold `GroupKFold` on training scans, grouping by object so scans of one training object stay in one fold. The selected alpha was 1000 (lowest mean held-out latent MSE: 6.890045).
5. Decode by either (a) clipping reconstructed occupancy scores and sampling 2048 points within occupied voxel cells, or (b) choosing the nearest training-object shape-PCA code and emitting that training point cloud. The latter is the validation-selected Iteration 2 configuration.

The occupancy decoder produces coarse voxel-cell point locations and shows low silhouette fidelity. The codebook decoder preserves the original training clouds and canonical coordinate system, but quantizes outputs to a small number of training shapes. Neither representation is a learned high-resolution surface decoder.

## Leakage safeguards

- Fitting reads only `train.npz` fMRI, groups, and object points; it does not load training images or masks.
- fMRI scaler/PCA, shape occupancy bounds/PCA/codebook, and Ridge weights use training data only.
- Ridge alpha selection is training-only and group-folded by the split-local training object IDs.
- Prediction reads only validation fMRI and sample IDs. Validation shapes/masks are first accessed by the supplied official evaluator; no validation target is used to fit a transform, shape codebook, or regression parameter.
- The train and validation `group` numbers index separate split-local tables. No equality comparison across splits is used.
- No test archive or test target data was used.

## Configuration, runtime, and reproducibility

The fitted checkpoint is `outputs/iteration2/reconstruction_model.joblib` (about 13.6 MB), with fitting settings in `outputs/iteration2/config.json`. The validation-selected decoder and metrics are also recorded in `outputs/iteration2/model_selection.json`. The selected, checkpoint-reproducible predictions are `outputs/iteration2/validation_predictions_best.npz`.

Observed fitting time was 6.1 seconds (including grouped alpha selection) on the available CPU. The reported process RSS was approximately 260 MiB after saving the checkpoint. The checkpoint regenerated identical codebook predictions in 1.2 seconds. Three instrumented official evaluator runs took 62.40 s (occupancy model), 75.76 s (shuffled codebook control), and 75.78 s (codebook model), approximately 214 seconds total.

Commands used (PowerShell, from the workspace root):

```powershell
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\train_reconstruction.py
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\train_reconstruction.py --predict-only --decoder codebook --prediction-name validation_predictions_codebook.npz
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\train_reconstruction.py --predict-only --decoder codebook --prediction-name validation_predictions_best.npz
& 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe' scripts\evaluate_iteration2.py
```

The training command is deterministic at the saved seed (17). Evaluation writes machine-readable JSON under `outputs/iteration2/`. Runtime packages used are listed in `requirements.txt`; no packages were installed during this iteration.

## Official validation comparison

All metrics below are from the unmodified `evaluate.py` on 125 validation scans. Composite is higher-is-better; 0 denotes the generic anchor and 1 the oracle retrieval anchor.

| Prediction | Chamfer (lower) | F-score (higher) | Silhouette IoU (higher) | Silhouette SSIM (higher) | Identification | Composite |
|---|---:|---:|---:|---:|---:|---:|
| Iteration 1 generic medoid | 0.1562 | 0.3947 | 0.4384 | 0.6919 | 48.76% | -0.013 |
| Iteration 1 direct fMRI nearest-scan retrieval | 0.2060 | 0.3253 | 0.3633 | 0.6417 | 49.48% | -0.158 |
| Iteration 2 occupancy decoder | 0.1622 | 0.4050 | 0.2590 | 0.2870 | 50.30% | -0.503 |
| Iteration 2 selected shape-codebook decoder | 0.2211 | 0.2659 | 0.2939 | 0.6844 | 52.59% | -0.141 |
| Selected codebook, shuffled scan association (one seed-17 permutation) | 0.2105 | 0.2870 | 0.3190 | 0.6920 | 57.60% | -0.055 |

The selected codebook decoder was the better of the two Iteration 2 decoders by validation composite (-0.141 versus -0.503), but its composite remains below both Iteration 1 baselines; the generic medoid remains the strongest overall baseline. Shuffling the selected outputs did not reduce identification: the shuffled control's identification and composite were higher. This is a negative result for scan-specific brain dependence, not evidence of successful reconstruction. The control uses one random permutation and is not a statistical significance test; no significance claim is made.

The selected model generated 22 unique clouds across 125 validation scans. Mean pointwise coordinate standard deviation across predictions was 0.133 in the codebook control. This confirms output variation, but variation by itself does not establish that the correct fMRI scan predicts its matching object.

## Failures and limitations

- Initial decoder coordinate-indexing errors were found by shape/finite checks, fixed, and the complete train/predict/evaluate path then passed.
- Validation metric results did not beat the generic medoid composite. Codebook identification is above 50% in the raw point estimate, but the shuffled control being higher argues against interpreting this as learned brain dependence.
- Occupancy quantization is coarse; pointwise sampling adds no learned surface detail. Shape PCA retains only 69.3% of occupancy variance and fMRI PCA is heavily dimensionality-reduced.
- The training set has only 292 distinct object clouds for 533 scans. The codebook retrieval decoder emits only 22 distinct validation shapes, so its nominal output variability is limited.
- Only Ridge alpha was selected with grouped training CV. PCA dimensionalities and voxel resolution were kept fixed; no broad hyperparameter search or repeated-shuffle uncertainty analysis was performed.
- A single shuffled control was evaluated; no p-value or confidence interval was calculated.

## Iteration 3 recommendation

Retain the generic medoid as the current strongest benchmark. For a meaningful next model, avoid a global 16x16x16 occupancy PCA as the only shape representation. First test a small, deterministic, permutation-invariant shape descriptor or a learned shape autoencoder using training shapes only. Then regress those shape latents from fMRI with group-held-out validation inside train, subject-aware regularization, and shuffled-fMRI controls. Evaluate multiple fixed seeds or a permutation test before claiming brain dependence. Continue reporting the generic medoid and direct retrieval as controls; do not treat the selected Iteration 2 checkpoint as a successful reconstruction model.
