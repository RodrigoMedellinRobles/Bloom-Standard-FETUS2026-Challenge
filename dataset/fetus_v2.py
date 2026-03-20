from copy import deepcopy
import h5py
import math
import json
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

from dataset.transform_v2 import (
    random_rot_flip, random_rotate, blur, obtain_cutmix_box,
    preprocess_no_zscore, zscore_normalize, zscore_normalize_tensor,
    augment_labeled_strong,
    augment_unlabeled_weak,
    augment_unlabeled_strong_np,
    speckle_noise, gamma_correction, random_brightness, random_contrast,
    coarse_dropout,
)


class FETUSSemiDataset(Dataset):
    """Semi-supervised dataset for Phase 1 (seg + cls).

    Modes: train_l, train_u, valid
    """

    def __init__(self, json_file_path, mode, size=None, n_sample=None,
                 oversample_positive=True, positive_repeat=3, strong_aug=True):
        self.json_file_path = json_file_path
        self.mode = mode
        self.size = size
        self.n_sample = n_sample
        self.strong_aug = strong_aug

        with open(self.json_file_path, mode='r') as f:
            self.case_list = json.load(f)

        if mode == 'train_l':
            if oversample_positive and positive_repeat > 1:
                original_len = len(self.case_list)
                positive_cases = []
                for case in self.case_list:
                    try:
                        with h5py.File(case['label'], 'r') as f:
                            label = f['label'][:]
                        if label.sum() > 0:
                            positive_cases.append(case)
                    except Exception:
                        pass

                n_pos = len(positive_cases)
                extra = positive_cases * (positive_repeat - 1)
                self.case_list = self.case_list + extra
                print(f"[FETUSSemiDataset] Oversampled: {original_len} -> "
                      f"{len(self.case_list)} (+{len(extra)} from {n_pos} positives)")

            if n_sample is not None:
                self.case_list *= math.ceil(n_sample / len(self.case_list))
                self.case_list = self.case_list[:n_sample]

    def __getitem__(self, item):
        case = self.case_list[item]

        if self.mode == 'valid':
            image_h5_file, label_h5_file = case['image'], case['label']
            with h5py.File(image_h5_file, mode='r') as f:
                image = preprocess_no_zscore(f['image'][:])
                image_view_id = int(np.array(f['view'][:]).reshape(-1)[0])
            with h5py.File(label_h5_file, mode='r') as f:
                mask = f['mask'][:]
                label = f['label'][:]

            image = zscore_normalize(image, image_view_id)

            return (torch.from_numpy(image).unsqueeze(0).float(),
                    torch.tensor(image_view_id - 1, dtype=torch.long),
                    torch.from_numpy(mask).long(),
                    torch.from_numpy(label).long())

        elif self.mode == 'train_l':
            image_h5_file, label_h5_file = case['image'], case['label']
            with h5py.File(image_h5_file, mode='r') as f:
                image = preprocess_no_zscore(f['image'][:])
                image_view_id = int(np.array(f['view'][:]).reshape(-1)[0])
            with h5py.File(label_h5_file, mode='r') as f:
                mask = f['mask'][:]
                label = f['label'][:]

            if self.strong_aug:
                image, mask = augment_labeled_strong(image, mask)
            else:
                if random.random() > 0.5:
                    image, mask = random_rot_flip(image, mask)
                elif random.random() > 0.5:
                    image, mask = random_rotate(image, mask)

            x, y = image.shape
            image = zoom(image, (self.size / x, self.size / y), order=0)
            mask = zoom(mask, (self.size / x, self.size / y), order=0)

            image = zscore_normalize(image, image_view_id)

            return (torch.from_numpy(image).unsqueeze(0).float(),
                    torch.tensor(image_view_id - 1, dtype=torch.long),
                    torch.from_numpy(mask).long(),
                    torch.from_numpy(label).long())

        elif self.mode == 'train_u':
            image_h5_file = case['image']
            with h5py.File(image_h5_file, mode='r') as f:
                image = preprocess_no_zscore(f['image'][:])
                image_view_id = int(np.array(f['view'][:]).reshape(-1)[0])

            image = augment_unlabeled_weak(image)

            x, y = image.shape
            image = zoom(image, (self.size / x, self.size / y), order=0)

            image_for_strong = image.copy()
            if self.strong_aug:
                image_for_strong = augment_unlabeled_strong_np(image_for_strong)

            image_pil = Image.fromarray((image_for_strong * 255).astype(np.uint8))
            image_s1, image_s2 = deepcopy(image_pil), deepcopy(image_pil)

            image = zscore_normalize(image, image_view_id)
            image = torch.from_numpy(image).unsqueeze(0).float()

            # Strong view 1
            if random.random() < 0.8:
                image_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_s1)
            image_s1 = blur(image_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
            image_s1 = torch.from_numpy(np.array(image_s1)).unsqueeze(0).float() / 255.0
            image_s1 = zscore_normalize_tensor(image_s1, image_view_id)

            # Strong view 2
            if random.random() < 0.8:
                image_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_s2)
            image_s2 = blur(image_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
            image_s2 = torch.from_numpy(np.array(image_s2)).unsqueeze(0).float() / 255.0
            image_s2 = zscore_normalize_tensor(image_s2, image_view_id)

            return (image, torch.tensor(image_view_id - 1, dtype=torch.long),
                    image_s1, image_s2, cutmix_box1, cutmix_box2)

    def __len__(self):
        return len(self.case_list)


class FETUSClsOnlyDataset(Dataset):
    """Classification-only dataset for Phase 2.

    Modes: train_l, train_u, valid
    """

    def __init__(self, json_file_path, mode='train_l', size=256, n_sample=None,
                 oversample_positive=True, positive_repeat=5):
        self.mode = mode
        self.size = size

        with open(json_file_path, 'r') as f:
            self.case_list = json.load(f)

        if mode == 'train_l':
            if oversample_positive:
                positive_cases = []
                for case in self.case_list:
                    try:
                        with h5py.File(case['label'], 'r') as f:
                            label = f['label'][:]
                        if label.sum() > 0:
                            positive_cases.append(case)
                    except Exception:
                        pass

                extra = positive_cases * (positive_repeat - 1)
                self.case_list = self.case_list + extra
                print(f"[ClsOnly] Oversampled: {len(self.case_list) - len(extra)} "
                      f"-> {len(self.case_list)} (+{len(extra)} positive copies)")

            if n_sample is not None:
                self.case_list *= math.ceil(n_sample / len(self.case_list))
                self.case_list = self.case_list[:n_sample]

    def __getitem__(self, item):
        case = self.case_list[item]

        if self.mode == 'train_l':
            image_h5_file, label_h5_file = case['image'], case['label']
            with h5py.File(image_h5_file, 'r') as f:
                image = preprocess_no_zscore(f['image'][:])
                view_id = int(np.array(f['view'][:]).reshape(-1)[0])
            with h5py.File(label_h5_file, 'r') as f:
                label = f['label'][:]

            if random.random() < 0.5:
                image = random_rot_flip(image)
            elif random.random() < 0.5:
                image = random_rotate(image)

            h, w = image.shape
            image = zoom(image, (self.size / h, self.size / w), order=0)

            if random.random() < 0.5:
                image = gamma_correction(image, (0.7, 1.5))
            if random.random() < 0.4:
                image = random_brightness(image, 0.12)
            if random.random() < 0.4:
                image = random_contrast(image, (0.75, 1.25))
            if random.random() < 0.3:
                image = speckle_noise(image, 0.04)
            if random.random() < 0.2:
                image = coarse_dropout(image, max_holes=4, max_size=20)

            image = zscore_normalize(image, view_id)

            return (torch.from_numpy(image).unsqueeze(0).float(),
                    torch.tensor(view_id - 1, dtype=torch.long),
                    torch.from_numpy(label).float())

        elif self.mode == 'train_u':
            image_h5_file = case['image']
            with h5py.File(image_h5_file, 'r') as f:
                image = preprocess_no_zscore(f['image'][:])
                view_id = int(np.array(f['view'][:]).reshape(-1)[0])

            if random.random() < 0.5:
                image = random_rot_flip(image)
            elif random.random() < 0.5:
                image = random_rotate(image)

            h, w = image.shape
            image = zoom(image, (self.size / h, self.size / w), order=0)

            image_strong = image.copy()
            if random.random() < 0.5:
                image_strong = gamma_correction(image_strong, (0.7, 1.5))
            if random.random() < 0.4:
                image_strong = random_brightness(image_strong, 0.12)
            if random.random() < 0.3:
                image_strong = speckle_noise(image_strong, 0.04)
            if random.random() < 0.2:
                image_strong = coarse_dropout(image_strong, max_holes=4, max_size=20)

            image_w = zscore_normalize(image, view_id)
            image_s = zscore_normalize(image_strong, view_id)

            return (torch.from_numpy(image_w).unsqueeze(0).float(),
                    torch.tensor(view_id - 1, dtype=torch.long),
                    torch.from_numpy(image_s).unsqueeze(0).float())

        elif self.mode == 'valid':
            image_h5_file, label_h5_file = case['image'], case['label']
            with h5py.File(image_h5_file, 'r') as f:
                image = preprocess_no_zscore(f['image'][:])
                view_id = int(np.array(f['view'][:]).reshape(-1)[0])
            with h5py.File(label_h5_file, 'r') as f:
                label = f['label'][:]

            image = zscore_normalize(image, view_id)

            return (torch.from_numpy(image).unsqueeze(0).float(),
                    torch.tensor(view_id - 1, dtype=torch.long),
                    torch.from_numpy(label).float())

    def __len__(self):
        return len(self.case_list)
