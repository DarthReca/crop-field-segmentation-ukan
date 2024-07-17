from typing import Callable, Literal

import numpy as np
import polars as pl
import rasterio as rio
import torch
from kornia.augmentation import (
    AugmentationSequential,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from lightning import LightningDataModule
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from stocaching import SharedCache
from torch import Tensor
from torchgeo.datasets import NonGeoDataset


class SACropTypeDataset(NonGeoDataset):
    rgb_bands = ["B04", "B03", "B02"]
    s1_bands = ["VH", "VV"]
    s2_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    all_bands: list[str] = s1_bands + s2_bands
    cmap = {
        0: (0, 0, 0, 255),
        1: (255, 211, 0, 255),
        2: (255, 37, 37, 255),
        3: (0, 168, 226, 255),
        4: (255, 158, 9, 255),
        5: (37, 111, 0, 255),
        6: (255, 255, 0, 255),
        7: (222, 166, 9, 255),
        8: (111, 166, 0, 255),
        9: (0, 175, 73, 255),
    }

    def __init__(
        self,
        path: str = "data",
        classes: list[int] = list(cmap.keys()),
        bands: list[str] = s2_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        mode: Literal["train", "val", "test"] = "train",
        binarize: bool = False,
    ) -> None:
        self.classes = classes
        self.bands = bands
        self.transforms = AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            data_keys=["image", "mask"],
        )
        self.mode = mode
        self.path = path
        self.binarize = binarize

        satellite = "s1" if bands[0] in self.s1_bands else "s2"
        self.df = (
            pl.read_parquet(f"{path}/{satellite}_split_iou.parquet")
            .filter(pl.col("split") == mode)
            .filter(pl.col("iou") < 0.3)
        )

        if mode in ("val", "test"):
            self.df = (
                self.df.filter(pl.col("saturated") == 0)
                .filter(pl.col("nulls") == 0)
                .filter(pl.col("iou") < 0.1)
                .group_by("area")
                .first()
            )

        self.areas = self.df["area"].unique().to_list()

        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=torch.int32)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)
        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

        self.cache = SharedCache(
            size_limit_gib=32,
            dataset_len=len(self),
            data_dims=(len(self.bands), 256, 256),
            dtype=torch.uint8,
        )

    def __len__(self) -> int:
        return self.df.unique("area").height

    def __getitem__(self, idx: int):
        area = self.areas[idx]
        date = self.df.filter(pl.col("area") == area).sample(n=1).item(0, "date")

        image = self.cache.get_slot(idx)
        if image is None:
            bands = []
            for b in self.bands:
                satellite = "s1" if b in self.s1_bands else "s2"
                with rio.open(
                    f"{self.path}/train/imagery/{satellite}/{area}/{date}/{area}_{date}_{b}_10m.tif"
                ) as src:
                    bands.append(src.read(1))
            image = torch.from_numpy(np.stack(bands))
            self.cache.set_slot(idx, image)

        with rio.open(f"{self.path}/train/labels/{area}.tif") as src:
            labels = src.read(1)
        labels = torch.from_numpy(labels)
        if self.binarize:
            labels[labels > 0] = 1

        if self.transforms is not None and self.mode == "train":
            image, labels = self.transforms(
                image.float().unsqueeze(0), labels.float().view(1, 1, 256, 256)
            )

        return image.float().squeeze(), labels.long().squeeze()

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        if not all(band in self.bands for band in self.rgb_bands):
            rgb_indices = [0]
        else:
            rgb_indices = [self.bands.index(band) for band in self.rgb_bands]

        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"].squeeze()
        ncols = 2

        showing_prediction = "prediction" in sample
        if showing_prediction:
            pred = sample["prediction"].squeeze()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs[0].imshow(image, cmap="gray" if len(rgb_indices) == 1 else None)
        axs[0].axis("off")
        axs[1].imshow(self.ordinal_cmap[mask], interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_prediction:
            axs[2].imshow(pred)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class SACropTypeDataModule(LightningDataModule):
    def __init__(
        self, path: str = "data", num_workers: int = 4, batch_size: int = 8, **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.path = path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.kwargs = kwargs

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            self.dataset_train = SACropTypeDataset(
                path=self.path, mode="train", **self.kwargs
            )
        if stage in ("fit", "val", None):
            self.val_dataset = SACropTypeDataset(
                path=self.path, mode="val", **self.kwargs
            )
        if stage in ("test", None):
            self.dataset_test = SACropTypeDataset(
                path=self.path, mode="test", **self.kwargs
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
