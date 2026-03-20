import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.transform_v2 import preprocess_no_zscore, zscore_normalize


class FETUSEvalDataset(Dataset):
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            self.case_list = json.load(f)

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx: int):
        case = self.case_list[idx]
        image_h5_file = case["image"]
        label_h5_file = case["label"]

        with h5py.File(image_h5_file, "r") as f:
            image = preprocess_no_zscore(f["image"][:])
            if "view" in f:
                view = int(np.array(f["view"][:]).reshape(-1)[0])
            else:
                view = 1

        with h5py.File(label_h5_file, "r") as f:
            mask = f["mask"][:]
            label = f["label"][:]

        image = zscore_normalize(image, view)

        image_t = torch.from_numpy(image).unsqueeze(0).float()
        view_t  = torch.tensor(view - 1, dtype=torch.long)
        mask_t  = torch.from_numpy(mask).long()
        label_t = torch.from_numpy(label).long()

        return image_t, view_t, mask_t, label_t, image_h5_file
