import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, retain_gd=None, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            retain_gd (Dataset, optional): Extra retain dataset for methods that
                need a separate gradient-descent retain stream.
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.retain_gd = retain_gd
        self.anchor = anchor
        self.datasets = {
            "forget": self.forget,
            "retain": self.retain,
            "retain_gd": self.retain_gd,
        }

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor not in self.datasets:
            raise NotImplementedError(
                f"{self.anchor} can only be one of {list(self.datasets.keys())}"
            )
        anchor_dataset = self.datasets[self.anchor]
        assert anchor_dataset is not None, ValueError(
            f"{self.anchor} dataset can't be None when anchor={self.anchor}"
        )
        return len(anchor_dataset)

    def __getitem__(self, idx):
        item = {self.anchor: self.datasets[self.anchor][idx]}
        for name, dataset in self.datasets.items():
            if name == self.anchor or dataset is None:
                continue
            sample_idx = torch.randint(0, len(dataset), (1,)).item()
            item[name] = dataset[sample_idx]
        return item
