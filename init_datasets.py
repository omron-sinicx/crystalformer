
from dataloaders.dataset_latticeformer import RegressionDatasetMP_Latticeformer as Dataset
from dataloaders.common import CellFormat
splits = ["train", "val", "test", "all"]
datasets = [
    "jarvis__megnet",
    "jarvis__megnet-shear",
    "jarvis__megnet-bulk",
    "jarvis__dft_3d_2021",
    "jarvis__dft_3d_2021-ehull",
    "jarvis__dft_3d_2021-mbj_bandgap",
]

import torch
for dataset in datasets:
    for split in splits:
        for format in [CellFormat.RAW, CellFormat.PRIMITIVE]:
            if ("shear" in dataset or "bulk" in dataset) and split == "all":
                continue
            print("Processing ------------------", dataset, split, format)
            data = Dataset(split, dataset, format)
            sizes = data.data.sizes.float()
            print(torch.mean(sizes).item(), torch.max(sizes).item(), torch.median(sizes).item(), torch.std(sizes).item())
