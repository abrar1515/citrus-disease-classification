# Notebook guide: `citrus_clean_pipeline.ipynb`

This guide describes the notebook in `notebooks/`, its settings, the files it writes, and how to resume an interrupted run. It complements the main `README.md`.

## 1. What the notebook does
One run executes the whole experiment and saves every metric, prediction, table and figure to disk, so nothing has to be re-run to regenerate a table or a plot.

1. Builds a manifest of the dataset (fruit and leaf images, six classes).
2. Creates the splits: a stratified 70/15/15 **hold-out** split and, optionally, stratified **five-fold cross-validation**.
3. Fine-tunes four ImageNet-pretrained CNNs (EfficientNetB0, ResNet50, DenseNet121, InceptionV3) with a two-phase procedure.
4. Trains SVM, Random Forest, KNN and Gaussian Naive Bayes on frozen ImageNet features of ResNet50 and DenseNet121 (and, optionally, on fine-tuned features).
5. Builds a soft-voting ensemble of the fine-tuned ResNet50 and DenseNet121 from the saved test probabilities.
6. Writes metrics, predictions, histories, LaTeX tables and 600 dpi figures (PNG + vector PDF), organ-wise accuracy, optional Grad-CAM, and the cross-validation analyses.

Every stage is cached: if its output files exist, the stage is skipped (`FORCE_RETRAIN = False`).

## 2. Requirements
- Developed and run on Kaggle (GPU, **Internet ON**, because Keras downloads the ImageNet weights).
- Environment of the reported run (see `results/run_info.json`): Python 3.13, TensorFlow 2.20, Keras 3.13, scikit-learn 1.6.1, two GPUs.
- Other libraries: NumPy, pandas, matplotlib, SciPy, Pillow, joblib. Optional: `psutil` (memory log lines).
- Dataset: Rauf et al. (2019), Mendeley Data, Version 2 (<https://data.mendeley.com/datasets/3f83gxmv57/2>). The notebook needs a folder that contains `Fruits/` and `Leaves/`, each with one sub-folder per class.

## 3. Configuration (the first code cell)
User switches:

| Switch | Default | Meaning |
|---|---|---|
| `SMOKE_TEST` | `False` | `True` = tiny synthetic dataset, random weights, 1+1 epochs, 2 folds. Only checks that every cell and output works. Its numbers are meaningless. |
| `FORCE_RETRAIN` | `False` | `True` ignores cached files and recomputes everything. |
| `RUN_CV` | `False` | `True` also runs the cross-validation (about `CV_FOLDS` times the hold-out time). |
| `CV_FOLDS` | `5` | Number of stratified folds. |
| `CV_RUN_FOLDS` | `None` | Train only these folds (1-based), for example `[4, 5]`. Finished folds are never recomputed. |
| `RESTORE_FROM` | `None` | Path of the `results` folder of an earlier run; it is copied into `OUT_DIR` before anything runs, so finished work is reused. |
| `RUN_FINETUNED_FEATURES` | `True` | Also train the classical classifiers on features of the fine-tuned CNNs. |
| `RUN_GRADCAM` | `True` | Grad-CAM figure for the final ResNet50 (inference only). |

Main entries of the `CFG` dictionary:

| Entry | Value |
|---|---|
| `SEED` | 42 (Python, NumPy, TensorFlow, scikit-learn, splits, generators) |
| `DATASET_DIR` | `None` = search `DATA_ROOT` (`/kaggle/input`) for a folder called `Citrus_dataset`; otherwise set the folder that contains `Fruits` and `Leaves` |
| `OUT_DIR` | `/kaggle/working/results` |
| `IMG` / `BATCH` | 224x224 / 32 for EfficientNetB0, ResNet50, DenseNet121; 299x299 / 24 for InceptionV3 |
| `DROPOUT` | 0.2 (EfficientNetB0), 0.4 (the other three) |
| `LR1`, `LR2` | 1e-3 (Phase 1, head only), 1e-5 (Phase 2, fine-tuning) |
| `EPOCHS1`, `EPOCHS2`, `N_UNFREEZE` | 20, 20, last 60 layers of the base |
| `ES_PATIENCE`, `RLR_FACTOR`, `RLR_PATIENCE`, `MIN_LR` | 8, 0.2, 8, 1e-6 (all on the validation loss) |
| `PCA_COMPONENTS` | 256 |
| `RF_TREES` | 600 (ResNet50 features), 500 (DenseNet121 features) |
| `FIG_DPI` | 600 |

## 4. Notebook sections

| Section | Content |
|---|---|
| 1 | Configuration and seed |
| 2 | Dataset manifest, class x organ counts |
| 3 | Hold-out split and class weights |
| 4 | Helpers: generators, model builder, metrics, saving |
| 5 | Two-phase training of one CNN (used for all four) |
| 6 | Feature extraction and classical classifiers |
| 7 | One experiment as one function (used for the hold-out run and for every fold) |
| 8 | Hold-out run |
| 9 | Figures and LaTeX tables of the hold-out run |
| 10 | Organ-wise accuracy (fruit vs leaf) |
| 11 | Grad-CAM |
| 12 | Cross-validation (optional) |
| 12b | Cross-validation analyses from the saved results (no training) |
| 13 | Run information, `README_outputs.txt`, download package |

Sections 9, 10 and 12b only read saved results. The cells before them (manifest and split) must run first; with the dataset attached they take seconds.

## 5. Method details

**Data and splits.** Both schemes are stratified by class with seed 42.
- Hold-out: 70% training, then the remaining 30% split equally into validation and test (531 / 114 / 114 images).
- Cross-validation: five stratified outer folds (test folds of 152, 152, 152, 152 and 151 images); in each round the remaining images are split, stratified, into training (about 515) and validation (92) images.
- Class weights `N / (K * n_c)` are computed from the training subset of each run.
- Augmentation (training images of the CNNs only): rotation 25 degrees, width/height shift 0.15, shear 0.10, zoom 0.20, horizontal flip, brightness [0.85, 1.15], channel shift 10, nearest fill. Images are normalised with the `preprocess_input` function of each architecture.

**Fine-tuning.** Head: global average pooling, dropout, dense softmax layer. The base always runs in inference mode (BatchNorm statistics are not updated). Phase 1 trains the head with the base frozen; Phase 2 unfreezes the last 60 layers. Each phase has its own checkpoint and the best-validation-loss weights are reloaded explicitly after each phase. The **final model is the phase with the lower best validation loss**. The Phase 1 model is also evaluated on the test data, so the effect of fine-tuning can be reported.

**Classical classifiers.** `StandardScaler -> PCA(256) -> classifier`, fitted on the training features only and evaluated once on the test features:
SVM (RBF, C=10, gamma=scale, balanced class weights), Random Forest (balanced-subsample class weights), KNN (k=5, distance weights), Gaussian Naive Bayes (var_smoothing 1e-3, training class frequencies as priors). Features come from the global-average-pooling output of the **frozen** ImageNet backbones (ResNet50: 2048, DenseNet121: 1024 dimensions) and optionally of the fine-tuned models.

**Ensemble.** The average of the softmax outputs of the fine-tuned ResNet50 and DenseNet121 (equal weights, no trainable parameters).

**Metrics.** Accuracy, macro precision, macro recall, macro F1 and weighted F1; precision of a class that is never predicted is counted as 0 (`zero_division=0`). Cross-validation results are reported as mean +/- SD (sample SD) over folds and as **pooled** results over the test predictions of all folds. Wilson 95% intervals and exact McNemar tests are computed on the pooled predictions; they treat the images as independent although the training sets of the folds overlap, so they are descriptive only.

## 6. Output files (`/kaggle/working/results`)
`results/README_outputs.txt` lists every file. In short:

| Folder / file | Content |
|---|---|
| `data/` | dataset manifest, class x organ counts, split manifests (hold-out and cross-validation), class weights |
| `holdout/` and `cv_fold1 ... cv_fold5/` | `metrics/` (one JSON per model with all metrics, per-class results and the confusion matrix; `summary.csv`), `preds/` (test predictions and probabilities), `history/` (training histories and `*_train_info.json` with epochs, best validation losses and the selected phase), `features/` and `models/` (not stored in this repository) |
| `cv_summary/` | `cv_per_fold_metrics.csv`, `cv_mean_sd.csv`, `cv_pooled_metrics.csv`, `cv_pooled_confusion.json`; `metrics/` (pooled per-class results, organ accuracy, ablation, Wilson intervals, McNemar tests); `tables/`; `figures/` |
| `paper_assets/` | figures (PNG at 600 dpi and vector PDF) and LaTeX `tabular` files used in the paper |
| `run_info.json`, `config.json` | software versions, GPU, seed, full configuration |

Model names in file names are made safe by replacing every non-alphanumeric character with an underscore, for example `ResNet50 (frozen) + SVM` becomes `ResNet50_frozen_SVM`. Fold models are deleted after each fold (`KEEP_CV_MODELS = False`) to save disk space.

## 7. Run time and memory
- Measured on Kaggle (two GPUs): the hold-out run took about 2,280 s (38 minutes) and each cross-validation fold about 2,200 to 2,360 s (37 to 39 minutes).
- In the reported run the first session ended in fold 4 with "Your notebook tried to allocate more memory than is available" after 19 trained models; the process memory grows with every trained model. The notebook prints `[mem]` lines after every model and fold.
- Recommendation: run the cross-validation in two or more sessions, for example `CV_RUN_FOLDS = [1, 2, 3]` and then `[4, 5]`, and restore the earlier results with `RESTORE_FROM`.

## 8. Resuming a run
1. Open the finished or failed notebook version on Kaggle and check in the Output tab that the folders (`holdout`, `cv_fold1`, ...) exist.
2. In the new session add that notebook output as an input (Add Input, Notebook Output) and find the path with `!find /kaggle/input -maxdepth 5 -type d -name cv_fold1`.
3. In the configuration cell set `RUN_CV = True`, `CV_FOLDS = 5`, `CV_RUN_FOLDS = [4, 5]` (the folds still missing) and `RESTORE_FROM = "<path to that results folder>"`.
4. *Run All* (or *Save & Run All*). Finished models are skipped. The cross-validation summary is built only from **complete** folds (a fold is complete when its ensemble result exists) and the notebook prints which folds were used.

## 9. Grad-CAM
Section 11 loads the final ResNet50 and explains the **stored** prediction for one correctly classified and one misclassified test image per class. Images are loaded exactly as in the data generators (nearest-neighbour resizing). The cell prints a sanity check, "re-computed predictions that differ from the stored predictions", which must be 0. The figure is qualitative only. This repository contains no Grad-CAM figure; re-run Section 11 and check that line before using it.

## 10. Troubleshooting

| Problem | Cause / fix |
|---|---|
| `Citrus_dataset not found` | Set `CFG["DATASET_DIR"]` to the folder that contains `Fruits` and `Leaves`. |
| Download error when building a model | Internet is OFF; switch it ON (ImageNet weights are downloaded). |
| `GPUs: []` | No accelerator selected; training would be very slow. |
| "Kernel died" / memory message | See Sections 7 and 8; run fewer folds per session. |
| Results look stale | The notebook warns if `CFG` differs from the cached run; set `FORCE_RETRAIN = True` or delete the output folder. |
| Different numbers after a re-run | GPU training is not bit-wise deterministic; differences of a few test images are expected. |

## 11. Smoke test
Set `SMOKE_TEST = True` and *Run All*. The notebook creates a tiny synthetic dataset and writes to `/tmp/citrus_smoke/results`, so your real outputs are not touched. It covers the hold-out run, all classical classifiers, the ensemble, all figures and tables, the organ analysis, Grad-CAM, two-fold cross-validation and the zip package. Afterwards set `SMOKE_TEST = False` and restart the session.