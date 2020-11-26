from datasets.img_dataset import FoodImgs
from models.cnn import SnapSnack
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import math
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import torch

dataset = FoodImgs(
    imgs_path="../input/snapandsnack/snapnsnackdb/simple_images",
    target_dict="../input/snapandsnack/snapnsnackdb/targets_dict.pkl",
)

s = len(dataset)
print(f"{s} imgs")
train_size = int(math.ceil(s * 0.8))
test_size = s - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])

wandb_logger = WandbLogger(project='snapnsnack')
model = SnapSnack()
trainer = pl.Trainer(logger=wandb_logger, gpus=1)
trainer.fit(model, DataLoader(train_set, batch_size=256), DataLoader(test_set))
torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
wandb.save(os.path.join(wandb.run.dir, "model.pt"))
