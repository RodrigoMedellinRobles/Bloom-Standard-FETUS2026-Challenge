import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.transform_v2 import preprocess_no_zscore, zscore_normalize


class FETUSInferDataset(Dataset):
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            self.case_list = json.load(f)

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx: int):
        case = self.case_list[idx]
        image_h5_file = case["image"]

        with h5py.File(image_h5_file, "r") as f:
            image = preprocess_no_zscore(f["image"][:])
            if "view" in f:
                view = int(np.array(f["view"][:]).reshape(-1)[0])
            else:
                view = 1

        image = zscore_normalize(image, view)

        image_t = torch.from_numpy(image).unsqueeze(0).float()
        view_t  = torch.tensor(view - 1, dtype=torch.long)

        return image_t, view_t, image_h5_file
