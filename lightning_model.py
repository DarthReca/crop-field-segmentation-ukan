from typing import Literal

import lightning as L
import matplotlib.pyplot as plt
import torch
from losses.gdice import GDiceLossV2
from models import UKAN, UNET
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import AdamW
from torchmetrics import F1Score, JaccardIndex, MetricCollection, Precision, Recall
from torchmetrics.wrappers import ClasswiseWrapper


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        model: Literal["UKAN", "UNET"] = "UKAN",
        num_classes=1,
        in_chans=3,
        img_size=256,
        lr=1e-4,
        loss="gdice",
    ):
        super().__init__()
        self.save_hyperparameters()

        embed_dims = [256, 512, 1024]
        if model == "UKAN":
            self.model = UKAN(num_classes, in_chans, img_size, embed_dims)
        else:
            self.model = UNET(n_classes=num_classes, n_channels=in_chans, bilinear=True)

        if loss == "gdice":
            self.loss = GDiceLossV2(
                self_compute_weight=True, apply_nonlin=Softmax(dim=1)
            )
        else:
            self.loss = CrossEntropyLoss()
        self.metric = ClasswiseWrapper(
            JaccardIndex(task="multiclass", num_classes=num_classes, average="none")
        )
        self.test_metric = MetricCollection(
            {
                "ji": ClasswiseWrapper(
                    JaccardIndex(
                        task="multiclass", num_classes=num_classes, average="none"
                    )
                ),
                "prec": ClasswiseWrapper(
                    Precision(
                        task="multiclass", num_classes=num_classes, average="none"
                    )
                ),
                "rec": ClasswiseWrapper(
                    Recall(task="multiclass", num_classes=num_classes, average="none")
                ),
                "f1": ClasswiseWrapper(
                    F1Score(task="multiclass", num_classes=num_classes, average="none")
                ),
            }
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.2, patience=5, min_lr=1e-10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x, y = batch["image"], batch["mask"]
        else:
            x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x, y = batch["image"], batch["mask"]
        else:
            x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.metric.update(y_hat, y)
        self.log("val_loss", loss)
        self.log_dict(self.metric.compute())

        if batch_idx < 10 and self.logger is not None:
            fig = self.trainer.datamodule.val_dataset.plot(
                {
                    "image": x[0].cpu().detach(),
                    "mask": y[0].cpu().detach(),
                    "prediction": y_hat[0].argmax(0).cpu().detach(),
                }
            )
            self.logger.experiment.log_figure(
                figure=fig, figure_name=f"val_{batch_idx}"
            )
            plt.close(fig)

        return loss

    def on_validation_epoch_end(self):
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x, y = batch["image"], batch["mask"]
        else:
            x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.test_metric.update(y_hat, y)
        self.log("test_loss", loss)
        self.log_dict(self.test_metric.compute())

    def on_test_epoch_end(self):
        self.test_metric.reset()
