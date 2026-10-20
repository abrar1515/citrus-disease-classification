# Citrus Disease Classification: Deep Learning, Classical ML and Hybrid Models

Comparison of fine-tuned CNNs, classical classifiers on deep features, and a soft-voting ensemble for classifying citrus fruit and leaf images into six classes (Black spot, Canker, Greening, Healthy, Melanose, Scab). Evaluation uses stratified five-fold cross-validation on all 759 images, plus one 70/15/15 hold-out split for comparison.

> Status: manuscript in preparation. This repository contains the notebook and the saved results behind the paper's tables and figures.

## Contents
- [Dataset](#dataset)
- [What is compared](#what-is-compared)
- [Main results](#main-results)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

## Dataset
Citrus Fruits and Leaves dataset by Rauf et al. (2019), Mendeley Data, Version 2: <https://data.mendeley.com/datasets/3f83gxmv57/2> (DOI 10.17632/3f83gxmv57.2). The images are **not** included in this repository; please download them from Mendeley Data and check the license on the dataset page.

759 images (256 x 256 px; 150 fruit and 609 leaf images). After merging fruit and leaf variants of each condition there are six classes:

| Class | Fruit | Leaf | Total |
|---|---:|---:|---:|
| Canker | 78 | 163 | 241 |
| Greening | 16 | 204 | 220 |
| Black spot | 19 | 171 | 190 |
| Healthy | 22 | 58 | 80 |
| Scab | 15 | 0 | 15 |
| Melanose | 0 | 13 | 13 |

Scab occurs only in fruit images and Melanose only in leaf images, so for these two classes the label is confounded with the organ type.

Expected folder layout (the notebook looks for a folder that contains `Fruits/` and `Leaves/`, each with one sub-folder per class):

```
<dataset_folder>
│
├── Fruits/
│   ├── Black spot/
│   ├── Canker/
│   ├── Greening/
│   ├── healthy/
│   └── Scab/
│
└── Leaves/
    ├── Black spot/
    ├── canker/
    ├── greening/
    ├── healthy/
    └── Melanose/
```

## What is compared
- **Fine-tuned CNNs:** EfficientNetB0, ResNet50, DenseNet121, InceptionV3 (ImageNet weights, global average pooling + dropout + softmax head, two-phase transfer learning: head only, then the last 60 layers; Adam; class-weighted loss; data augmentation; early stopping, learning-rate reduction and checkpointing on the validation loss).
- **DL+ML hybrids:** frozen ImageNet ResNet50 and DenseNet121 features -> StandardScaler -> PCA (256) -> SVM (RBF), Random Forest, KNN or Gaussian Naive Bayes (fitted on the training split only).
- **DL+DL hybrid:** equal-weight soft-voting ensemble of the fine-tuned ResNet50 and DenseNet121.
- **Extra analyses:** Phase 1 (frozen) vs final model, frozen vs fine-tuned features, accuracy by organ (fruit vs leaf), pooled per-class results, Wilson intervals and McNemar tests.
- **Metrics:** accuracy, macro precision/recall/F1, weighted F1 (undefined precision counted as 0).

## Main results
Accuracy (%) in five-fold cross-validation (mean +/- SD over the folds; each fold has 151-152 test images) and on the single hold-out split (114 test images):

| Model | CV accuracy | Hold-out accuracy |
|---|---:|---:|
| ResNet50 + SVM (frozen features) | 87.9 +/- 3.7 | 93.0 |
| ResNet50 (fine-tuned) | 87.4 +/- 2.7 | 88.6 |
| DenseNet121 + SVM (frozen features) | 87.4 +/- 2.7 | 92.1 |
| ResNet50 + DenseNet121 (soft voting) | 86.7 +/- 3.3 | 91.2 |
| EfficientNetB0 | 82.4 +/- 4.7 | 88.6 |
| DenseNet121 | 82.4 +/- 2.9 | 85.1 |
| DenseNet121 + KNN | 81.8 +/- 1.9 | 82.5 |
| InceptionV3 | 81.2 +/- 3.9 | 86.8 |
| ResNet50 + RF | 79.0 +/- 3.0 | 81.6 |
| ResNet50 + KNN | 78.5 +/- 3.9 | 79.8 |
| DenseNet121 + RF | 78.4 +/- 0.6 | 81.6 |
| DenseNet121 + NB | 62.8 +/- 4.2 | 61.4 |
| ResNet50 + NB | 45.8 +/- 5.7 | 41.2 |

Main findings:
- The four leading models (87.9, 87.4, 87.4 and 86.7 %) cannot be distinguished within the fold-to-fold variation (pooled McNemar tests, p between 0.34 and 1.0).
- The single hold-out split was favourable: it gave a higher accuracy than the cross-validation mean for 11 of 13 models, by up to 6 points, and ranked the SVMs clearly above the fine-tuned networks, which the cross-validation did not confirm.
- Accuracy was lower on fruit images (about 69-75 % for the leading models) than on leaf images (about 91 %).
- All numbers, per-class results and statistics are in `results/`.

## Repository structure
```
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── citrus_clean_pipeline.ipynb      # full pipeline (no outputs stored)
├── docs/
│   └── NOTEBOOK_GUIDE.md                # details of the notebook, outputs and resuming a run
└── results/
    ├── README_outputs.txt               # description of every output file
    ├── run_info.json, config.json       # software versions, GPU, seed, configuration
    ├── data/                            # dataset manifest, class x organ counts, split manifests, class weights
    ├── holdout/                         # metrics/, preds/, history/ of the 70/15/15 run
    ├── cv_fold1 ... cv_fold5/           # metrics/, preds/, history/ of each fold
    ├── cv_summary/                      # mean/SD, pooled metrics, metrics/, tables/, figures/
    └── paper_assets/                    # figures (600 dpi PNG + PDF) and LaTeX tables
```
Large files (trained models `*.keras`, extracted features `*.npz`) are not stored; the notebook recreates them.

## How to run
The notebook was developed for Kaggle (GPU, Internet ON because Keras downloads the ImageNet weights).

1. Upload `notebooks/citrus_clean_pipeline.ipynb` and attach the citrus dataset.
2. In the configuration cell set `DATASET_DIR` to the folder that contains `Fruits` and `Leaves`, or leave it `None` if the folder is called `Citrus_dataset`.
3. Optional dry run: `SMOKE_TEST = True` uses a tiny synthetic dataset and random weights (about 8 minutes on CPU, numbers are meaningless) and checks that every cell and output works. Set it back to `False` afterwards.
4. Hold-out run: `RUN_CV = False`, then *Run All*. Cross-validation: `RUN_CV = True` (about 5 times the training time).
5. Outputs are written to `/kaggle/working/results`; download `results_package.zip`.

Resuming after an interrupted run: every finished model is cached. Use `RESTORE_FROM` (copy the results of an earlier run into the output folder) and `CV_RUN_FOLDS` (for example `[4, 5]`) to run only some folds; see `docs/NOTEBOOK_GUIDE.md`.

## Reproducibility notes
- Fixed seed 42 for the splits, data generators, TensorFlow/NumPy and scikit-learn. GPU training is not fully deterministic, so repeated runs can differ by a few test images.
- Environment of the reported run (`results/run_info.json`): Python 3.13, TensorFlow 2.20, Keras 3.13, scikit-learn 1.6.1, two GPUs.
- The cross-validation was run in two sessions because the first one ran out of host memory in fold 4; finished results were reused with the same code and configuration.
- Hold-out/cross-validation numbers in the paper come from this notebook; nothing is copied from earlier notebooks.

## Limitations
Small and imbalanced dataset (13 Melanose and 15 Scab images); label-organ confounding for Scab and Melanose; possible near-duplicate images across folds (the dataset gives no tree or fruit identifiers); one cross-validation run with one seed; no per-model hyperparameter tuning; no inference-time or field-image evaluation.

## Citation
If you use this code or the results, please cite the manuscript (in preparation) and the dataset:

```bibtex
@article{saddique_citrus_2026,
  title  = {Robust Automated Diagnosis of Citrus Fruits and Leaves Diseases Using Hybrid Deep Learning and Machine Learning Integration},
  author = {Saddique, Muhammad Abrar},
  note   = {Manuscript in preparation}
}

@article{rauf2019citrus,
  title   = {A citrus fruits and leaves dataset for detection and classification of citrus diseases through machine learning},
  author  = {Rauf, Hafiz Tayyab and Saleem, Basharat Ali and Lali, M. Ikram Ullah and Khan, Muhammad Attique and Sharif, Muhammad and Bukhari, Syed Ahmad Chan},
  journal = {Data in Brief},
  volume  = {26},
  pages   = {104340},
  year    = {2019},
  doi     = {10.1016/j.dib.2019.104340}
}
```

## License
Code: MIT License (see `LICENSE`). The dataset has its own license on its Mendeley Data page and is not redistributed here.

## Links
- Kaggle notebook: <ADD LINK>
- Paper / preprint: <ADD LINK WHEN AVAILABLE>