# Fetus2026 Challenge — 1st Place Solution

Semi-supervised fetal cardiac structure segmentation and congenital heart disease (CHD) classification from prenatal ultrasound images.

**Challenge**: [Fetus2026](http://119.29.231.17:90/index.html) — Fetal HearT UltraSound Segmentation and Diagnosis  
**Score formula**: `0.45 * F1 + 0.45 * (DSC + NSD) / 2 + 0.1 * T-score`

## Overview

The dataset contains ~2800 fetal cardiac ultrasound images across 4 standard views (4CH, LVOT, RVOT, 3VT), of which ~391 have segmentation masks and classification labels. The rest are unlabeled. Two tasks:

1. **Segmentation** — 14 anatomical classes (LA, LV, RA, RV, WholeHeart, DescAo, Thorax, AscAo, MainPA, LPA, RPA, SVC, AoArch, Trachea) + background
2. **Classification** — 7 binary CHD labels: VSD, AV_sten, AoHypo, AoV_sten, DORV, PV_sten, RAA

Each view only allows certain structures and diseases (view-based masking), which is critical for both training and inference.

## Architecture

ResNet-50 (ImageNet V2) encoder with a UNETR-style decoder, deep supervision at 4 decoder levels, and a boundary detection head that fuses edge features back into the segmentation path.

For classification, the model uses multi-scale attention pooling over enc3 (256-dim), enc4 (512-dim) and dec4 (1024-dim), concatenated into a 1792-dim vector fed to the classification MLP. A MorphHead extracts soft anatomical ratios from the segmentation output (LV/RV balance, AscAo/PA ratio, etc.) and adds gated logit corrections for AoHypo, AoV_sten and DORV — diseases with clear morphological signatures.

## Training

Two-phase approach:

**Phase 1** — Joint segmentation + classification with UniMatch semi-supervised learning. The labeled set (~240-390 images depending on split) trains with CE + Dice + boundary loss + deep supervision. The unlabeled set (~2900 images) trains with consistency regularization, feature perturbation, and CutMix. Classification uses focal BCE with view masking and pos_weight correction.

**Phase 2** — Classification-only fine-tuning. Segmentation weights are frozen. The classification head (attention pools + MLP + MorphHead) is re-initialized and trained with ASL loss (gamma_pos=0, gamma_neg=4, clip=0.05), same-view MixUp, and self-teaching pseudo-labels on unlabeled data. The model generates its own pseudo-labels in eval mode rather than using an EMA teacher, which we found reduces confirmation bias and false positive rates significantly.

## Inference

- 2-flip (or 4-flip) TTA on segmentation
- View-based logit masking
- Connected component deduplication (flat neighbor voting)
- Per-class sigmoid thresholds (manually calibrated)
- Max-1 disease per patient post-processing
- CDMAD: blank-image logit debiasing to revive collapsed classes

## Project Structure

```
dataset/
    transform_v2.py          Preprocessing + augmentation pipeline
    fetus_v2.py              Semi-supervised and cls-only datasets
    fetus_eval.py            Evaluation dataset (with labels)
    fetus_infer.py           Inference dataset (no labels)

model/
    echocare_resnet.py       Single model: ResNet + UNETR + DS + Boundary + AttPool + MorphHead

src/
    utils.py                 Losses, metrics, NSD, view masking, checkpoint utils

01a_data_splitting_dev.ipynb              Data split with 150-sample validation
01b_data_splitting_submission.ipynb       Data split for submission (all labeled -> train)
02a_train_phase1_dev.ipynb                Phase 1 training with validation
02b_train_phase1_submission.ipynb         Phase 1 training without validation
03a_train_phase2_dev.ipynb                Phase 2 cls fine-tuning with validation
03b_train_phase2_submission.ipynb         Phase 2 cls fine-tuning + inference + submission
```

The `*a` notebooks are the development pipeline (held-out validation for monitoring metrics and tuning decisions). The `*b` notebooks are the final submission pipeline (all labeled data for training, checkpoint selection via separation metric post-training). Both pipelines use the same model and dataset code — the only difference is whether data is held out for validation.

## Preprocessing

```
rgb2gray -> percentile clip (1%, 99%) -> CLAHE (2.0, 8x8) -> per-view z-score
```

Per-view normalization stats are computed from the training labeled set only. All images resized to 256x256 for training.

## View-Class Constraints

Segmentation:
| View | Allowed classes |
|------|----------------|
| 4CH  | BG, LA, LV, RA, RV, WholeHeart, DescAo, Thorax |
| LVOT | BG, LA, LV, RV, AscAo |
| RVOT | BG, DescAo, AscAo, MainPA, LPA, RPA, SVC |
| 3VT  | BG, MainPA, SVC, AoArch, Trachea |

Classification:
| View | Allowed diseases |
|------|-----------------|
| 4CH  | VSD, AV_sten |
| LVOT | VSD, AoHypo, AoV_sten |
| RVOT | DORV, PV_sten |
| 3VT  | AoHypo, PV_sten, RAA |

## Key Findings

- **Self-teaching beats EMA**: The EMA teacher inflated false positives through confirmation bias. Letting the model generate pseudo-labels in eval mode (self-teaching) cut FPs dramatically.
- **ASL is essential for rare classes**: PV_sten and other low-prevalence diseases were completely dead with standard BCE. ASL (gamma_neg=4) was the only technique that reliably activated them.
- **CDMAD fixes collapsed classes**: Subtracting blank-image logit biases before inference revived classes like AoHypo from 0 predictions to correct detection, without aggressive threshold changes.
- **Threshold calibration on training data is unreliable**: Produces inflated thresholds (0.93+). Manual tuning with budget-based reasoning on test set ratios worked better in practice.
- **ResNet generalizes better than Swin**: Swin had ~60% performance drop from local to platform evaluation. ResNet dropped ~22% and was consistently more stable.
- **Dense CRF doesn't work on ultrasound**: Speckle noise confuses the bilateral kernel, causing catastrophic metric degradation.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- MONAI
- torchvision
- scikit-learn, scikit-image
- scipy, opencv-python, h5py
- tqdm, matplotlib, tensorboard

## Data Setup

The challenge data should be organized as:
```
project_root/
    train/
        images/    (1.h5, 2.h5, ..., 2800.h5)
        labels/    (115_label.h5, 122_label.h5, ...)
    valid/
        images/    (challenge validation, 500 images)
```

Run `01a` or `01b` from the notebooks directory (one level inside project_root) to generate the JSON splits and preprocessing stats.
