# Fetus2026 Challenge — 1st Place Solution

Semi-supervised fetal cardiac structure segmentation and congenital heart disease (CHD) classification from prenatal ultrasound images.

**Challenge**: [Fetus2026](http://119.29.231.17:90/index.html) — Fetal HearT UltraSound Segmentation and Diagnosis  
**Score**: `0.45 \* F1 + 0.45 \* (DSC + NSD) / 2 + 0.1 \* T-score`

## Overview

The dataset contains \~2800 fetal cardiac ultrasound images across 4 standard views (4CH, LVOT, RVOT, 3VT), of which 391 have segmentation masks and classification labels. The rest are unlabeled. Two tasks:

1. **Segmentation** — 14 anatomical classes (LA, LV, RA, RV, WholeHeart, DescAo, Thorax, AscAo, MainPA, LPA, RPA, SVC, AoArch, Trachea) + background
2. **Classification** — 7 binary CHD labels: VSD, AV\_sten, AoHypo, AoV\_sten, DORV, PV\_sten, RAA

Each cardiac view only allows certain structures and diseases (view-based masking), which is enforced during both training and inference.

## Architecture

ResNet-50 (ImageNet V2) encoder with a UNETR-style decoder, deep supervision at 4 decoder levels, and a boundary detection head that fuses edge features back into the segmentation path.

For classification, multi-scale attention pooling over enc3 (256-dim), enc4 (512-dim) and dec4 (1024-dim), concatenated into a 1792-dim vector fed to the classification MLP. A MorphHead extracts soft anatomical ratios from the segmentation output (LV/RV balance, AscAo/PA ratio, etc.) and adds gated logit corrections for AoHypo, AoV\_sten and DORV.

Everything lives in a single model class (`Echocare\_ResNet`) used for both training phases.

## Training

**Phase 1** — Joint segmentation + classification with UniMatch semi-supervised learning. CE + Dice + boundary loss + deep supervision on labeled data. Consistency regularization, feature perturbation, and CutMix on unlabeled data. Classification uses focal BCE with view masking and pos\_weight correction.

**Phase 2** — Classification-only fine-tuning. Segmentation weights are frozen. The classification head (attention pools + MLP + MorphHead) is re-initialized and trained with ASL loss (gamma\_pos=0, gamma\_neg=4, clip=0.05), same-view MixUp, and self-teaching pseudo-labels on unlabeled data.

## Inference

Single forward pass per image. No TTA, no post-processing, no ensemble:

1. Resize to 256x256, forward through model
2. Segmentation: upsample logits to original size, apply view mask, argmax
3. Classification: sigmoid, apply view mask, per-class threshold
4. Save as `.h5` with `mask` (uint8 512x512) and `label` (uint8 7,)

Thresholds are manually calibrated per class.

## Project Structure

```
dataset/
    transform\_v2.py              Preprocessing + augmentation pipeline
    fetus\_v2.py                  Semi-supervised and cls-only datasets
    fetus\_eval.py                Evaluation dataset (with labels)
    fetus\_infer.py               Inference dataset (no labels)

model/
    echocare\_resnet.py           Single model for both phases

src/
    utils.py                     Losses, metrics, NSD, view masking, checkpoint utils

01\_data\_splitting.ipynb     Data split with 150-sample validation
02\_train\_phase1.ipynb       Phase 1: UniMatch semi-supervised (seg + cls)
03\_train\_phase2.ipynb       Phase 2: cls fine-tuning + inference
```

## Preprocessing

```
rgb2gray -> percentile clip (1%, 99%) -> CLAHE (2.0, 8x8) -> per-view z-score
```

Stats computed from training labeled set only. All images resized to 256x256.

## View-Class Constraints

Segmentation:

|View|Allowed classes|
|-|-|
|4CH|BG, LA, LV, RA, RV, WholeHeart, DescAo, Thorax|
|LVOT|BG, LA, LV, RV, AscAo|
|RVOT|BG, DescAo, AscAo, MainPA, LPA, RPA, SVC|
|3VT|BG, MainPA, SVC, AoArch, Trachea|

Classification:

|View|Allowed diseases|
|-|-|
|4CH|VSD, AV\_sten|
|LVOT|VSD, AoHypo, AoV\_sten|
|RVOT|DORV, PV\_sten|
|3VT|AoHypo, PV\_sten, RAA|

## Requirements

* Python >= 3.10
* PyTorch >= 2.0
* MONAI
* torchvision
* scikit-learn, scikit-image
* scipy, opencv-python, h5py
* tqdm, matplotlib, tensorboard

## Data Setup

```
project\_root/
    train/
        images/    (1.h5, 2.h5, ..., 2800.h5)
        labels/    (115\_label.h5, 122\_label.h5, ...)
    valid/
        images/    (challenge validation, 500 images)
```

Run `01\_data\_splitting.ipynb` from the project root to generate JSON splits and preprocessing stats.

