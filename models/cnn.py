import torch
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F
import wandb
from sklearn.metrics import r2_score
import numpy as np
from ignite.contrib.metrics.regression import R2Score

class SnapSnack(pl.LightningModule):

    def __init__(self, fc_layers=(256, 32), output_dim=5, lr=1, weight_decay=0.001):
        super().__init__()
        self.backbone = resnet18()
        layers = []
        prev = 512
        for i, dim in enumerate(fc_layers):
            layers.append(torch.nn.Linear(in_features = prev, out_features=dim))
            # if i % 2==0:
            #     layers.append(torch.nn.ReLU())
            # else:
            layers.append(torch.nn.Softmax())
            prev = dim
        layers.append(torch.nn.Linear(in_features=prev, out_features=output_dim))
        self.softmax = torch.nn.Tanh()
        self.backbone.fc = torch.nn.Sequential(*layers)
        self.loss = torch.nn.SmoothL1Loss

        self.lr = lr
        self.prev_preds = None
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.backbone(x)
        # x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self.forward(imgs)
        loss = F.mse_loss(preds, targets)
        if self.prev_preds is None:
            self.prev_preds = preds
        else:
            if torch.all(self.prev_preds == preds):
                print(preds, targets)
                print(preds[:, 0], targets[:, 0])
                print("=======================")
            self.prev_preds = preds

        r2_calories = r2_score(
            targets[:, 0].view(-1).cpu().detach().numpy(),
            preds[:, 0].view(-1).cpu().detach().numpy()
        )
        r2_proteins = r2_score(
            targets[:, 1].view(-1).cpu().detach().numpy(),
            preds[:, 1].view(-1).cpu().detach().numpy()
        )
        r2_fat = r2_score(
            targets[:, 2].view(-1).cpu().detach().numpy(),
            preds[:, 2].view(-1).cpu().detach().numpy()
        )
        r2_sodium = r2_score(
            targets[:, 3].view(-1).cpu().detach().numpy(),
            preds[:, 3].view(-1).cpu().detach().numpy()
        )
        r2_overall = r2_score(
            targets.cpu().detach().numpy(),
            preds.cpu().detach().numpy(),
        )
        self.log('train_loss', loss)
        log_obj = dict(
            loss=loss,
            batch_nb=batch_idx,
            r2_calories=r2_calories,
            r2_proteins=r2_proteins,
            r2_fat=r2_fat,
            r2_sodium=r2_sodium,
            r2_overall=r2_overall,
        )
        wandb.log(log_obj)
        return log_obj

    def training_epoch_end(self, outputs):
        calories = np.mean([x["r2_calories"] for x in outputs])
        protein = np.mean([x["r2_proteins"] for x in outputs])
        fat = np.mean([x["r2_fat"] for x in outputs])
        sodium = np.mean([x["r2_sodium"] for x in outputs])
        overall = np.mean([x["r2_overall"] for x in outputs])
        avg_loss = np.mean([x["loss"] for x in outputs])

        wandb.log({
            "epoch_overall_r2" : overall,
            "epoch_calories_r2": calories,
            "epoch_protein_r2": protein,
            "epoch_fat_r2": fat,
            "epoch_sodium_r2": sodium,
        })


    def validation_step(self, batch, batch_nb):
        imgs, targets = batch
        preds = self.forward(imgs)
        loss = F.mse_loss(preds, targets)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        wandb.log(dict(avg_val_loss=avg_loss.item()))
        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        imgs, targets = batch
        preds = self.forward(imgs)
        loss = F.mse_loss(preds, targets)
        return {'test_loss': loss.item()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
