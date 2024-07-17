from datetime import datetime

import comet_ml
import hydra
import lightning as L
import torch
from datasets import SACropTypeDataModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from lightning_model import SegmentationModel
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    dm = SACropTypeDataModule(**cfg.datamodule)
    model = SegmentationModel(**cfg.model)

    experiment_name = f"{model.model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger = False
    if cfg.log_comet:
        logger = CometLogger(
            save_dir="comet_logs",
            project_name="kan-for-eo",
            experiment_name=experiment_name,
        )
    callbacks = [
        ModelCheckpoint(
            f"checkpoints/{experiment_name}",
            monitor="val_loss",
            save_top_k=2,
            save_last=True,
        ),
        LearningRateMonitor(),
    ]

    trainer = L.Trainer(
        callbacks=callbacks if logger else None,
        logger=logger,
        precision="32-true",
        **cfg.trainer,
    )
    trainer.fit(model=model, datamodule=dm)
    logger.experiment.end()


if __name__ == "__main__":
    main()
