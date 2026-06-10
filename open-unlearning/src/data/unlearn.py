import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(
        self,
        forget,
        retain,
        retain_gd=None,
        anchor="forget",
        batch_mode="paired",
        batch_order="random",
    ):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            retain_gd (Dataset, optional): Extra retain dataset for methods that
                need a separate gradient-descent retain stream.
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
            batch_mode (str, optional): "paired" samples non-anchor datasets with
                replacement in each item. "unpaired" exposes each split item once in
                homogeneous scheduled batches. "mixed" exposes each split item once
                and lets batches contain multiple split types.
            batch_order (str, optional): For unpaired mode, one of "random",
                "forget_first", or "retain_first".
        """
        self.forget = forget
        self.retain = retain
        self.retain_gd = retain_gd
        self.anchor = anchor
        self.batch_mode = batch_mode
        self.batch_order = batch_order
        self.datasets = {
            "forget": self.forget,
            "retain": self.retain,
            "retain_gd": self.retain_gd,
        }
        self.active_datasets = {
            name: dataset for name, dataset in self.datasets.items() if dataset is not None
        }

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.batch_mode in ("unpaired", "mixed"):
            return sum(len(dataset) for dataset in self.active_datasets.values())

        if self.anchor not in self.datasets:
            raise NotImplementedError(
                f"{self.anchor} can only be one of {list(self.datasets.keys())}"
            )
        anchor_dataset = self.datasets[self.anchor]
        assert anchor_dataset is not None, ValueError(
            f"{self.anchor} dataset can't be None when anchor={self.anchor}"
        )
        return len(anchor_dataset)

    def _unpaired_index(self, idx):
        for name in ("forget", "retain", "retain_gd"):
            dataset = self.datasets.get(name)
            if dataset is None:
                continue
            if idx < len(dataset):
                return name, idx
            idx -= len(dataset)
        raise IndexError(idx)

    def __getitem__(self, idx):
        if self.batch_mode in ("unpaired", "mixed"):
            name, local_idx = self._unpaired_index(idx)
            if self.batch_mode == "mixed":
                item = dict(self.datasets[name][local_idx])
                item["_split"] = name
                return item
            return {name: self.datasets[name][local_idx]}

        item = {self.anchor: self.datasets[self.anchor][idx]}
        for name, dataset in self.datasets.items():
            if name == self.anchor or dataset is None:
                continue
            sample_idx = torch.randint(0, len(dataset), (1,)).item()
            item[name] = dataset[sample_idx]
        return item

    def get_scheduled_indices(self, batch_size):
        if self.batch_mode not in ("unpaired", "mixed"):
            return None

        groups = {}
        offset = 0
        for name in ("forget", "retain", "retain_gd"):
            dataset = self.datasets.get(name)
            if dataset is None:
                continue
            groups[name] = list(range(offset, offset + len(dataset)))
            offset += len(dataset)

        def chunks(indices):
            return [
                indices[start : start + batch_size]
                for start in range(0, len(indices), batch_size)
            ]

        if self.batch_mode == "mixed":
            indices = [idx for group in groups.values() for idx in group]
            if self.batch_order == "random":
                permutation = torch.randperm(len(indices)).tolist()
                return [indices[i] for i in permutation]
            if self.batch_order == "forget_first":
                return groups.get("forget", []) + groups.get("retain", []) + groups.get("retain_gd", [])
            if self.batch_order == "retain_first":
                return groups.get("retain", []) + groups.get("forget", []) + groups.get("retain_gd", [])
            raise NotImplementedError(
                f"{self.batch_order} must be one of random, forget_first, retain_first"
            )

        group_chunks = {name: chunks(indices) for name, indices in groups.items()}
        if self.batch_order == "forget_first":
            ordered_chunks = group_chunks.get("forget", []) + group_chunks.get("retain", [])
        elif self.batch_order == "retain_first":
            ordered_chunks = group_chunks.get("retain", []) + group_chunks.get("forget", [])
        elif self.batch_order == "random":
            ordered_chunks = group_chunks.get("forget", []) + group_chunks.get("retain", [])
            permutation = torch.randperm(len(ordered_chunks)).tolist()
            ordered_chunks = [ordered_chunks[i] for i in permutation]
        else:
            raise NotImplementedError(
                f"{self.batch_order} must be one of random, forget_first, retain_first"
            )

        if "retain_gd" in group_chunks:
            ordered_chunks += group_chunks["retain_gd"]
        return [idx for chunk in ordered_chunks for idx in chunk]

    def get_scheduled_batches(self, batch_size, drop_last=False):
        if self.batch_mode not in ("unpaired", "mixed"):
            return None

        def maybe_drop(batches):
            if drop_last:
                return [batch for batch in batches if len(batch) == batch_size]
            return batches

        if self.batch_mode == "mixed":
            indices = self.get_scheduled_indices(batch_size)
            batches = [
                indices[start : start + batch_size]
                for start in range(0, len(indices), batch_size)
            ]
            return maybe_drop(batches)

        groups = {}
        offset = 0
        for name in ("forget", "retain", "retain_gd"):
            dataset = self.datasets.get(name)
            if dataset is None:
                continue
            groups[name] = list(range(offset, offset + len(dataset)))
            offset += len(dataset)

        def chunks(indices):
            return [
                indices[start : start + batch_size]
                for start in range(0, len(indices), batch_size)
            ]

        group_chunks = {name: chunks(indices) for name, indices in groups.items()}
        if self.batch_order == "forget_first":
            batches = group_chunks.get("forget", []) + group_chunks.get("retain", [])
        elif self.batch_order == "retain_first":
            batches = group_chunks.get("retain", []) + group_chunks.get("forget", [])
        elif self.batch_order == "random":
            batches = group_chunks.get("forget", []) + group_chunks.get("retain", [])
            permutation = torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in permutation]
        else:
            raise NotImplementedError(
                f"{self.batch_order} must be one of random, forget_first, retain_first"
            )

        if "retain_gd" in group_chunks:
            batches += group_chunks["retain_gd"]
        return maybe_drop(batches)
