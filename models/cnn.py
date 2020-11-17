import torch
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F
import wandb

class SnapSnack(pl.LightningModule):

    def __init__(self, fc_layers=(256, 32), output_dim=5 ):
        super().__init__()
        self.backbone = resnet18()
        layers = []
        prev = 512
        for dim in fc_layers:
            layers.append(torch.nn.Linear(in_features = prev, out_features=dim))
            layers.append(torch.nn.ReLU())
            prev = dim
        layers.append(torch.nn.Linear(in_features=prev, out_features=output_dim))
        # self.softmax = torch.nn.Softmax(dim=1)

        self.backbone.fc = torch.nn.Sequential(*layers)
        self.loss = torch.nn.SmoothL1Loss

    def forward(self, x):
        x = self.backbone(x)
        # x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        imgs, targets = batch
        preds = self.forward(imgs)
        loss = F.smooth_l1_loss(preds, targets)

        self.log('train_loss', loss)
        wandb.log(dict(
            loss=loss.item(),
            batch_nb=batch_idx,
        ))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-7)
        return optimizer
