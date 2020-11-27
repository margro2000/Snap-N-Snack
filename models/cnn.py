import torch
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F
import wandb
from sklearn.metrics import r2_score

class SnapSnack(pl.LightningModule):

    def __init__(self, fc_layers=(256, 32), output_dim=5 ):
        super().__init__()
        self.backbone = resnet18()
        layers = []
        prev = 512
        for dim in fc_layers:
            layers.append(torch.nn.Linear(in_features = prev, out_features=dim))
            layers.append(torch.nn.Softmax(dim=1))
            prev = dim
        layers.append(torch.nn.Linear(in_features=prev, out_features=output_dim))
        self.softmax = torch.nn.Tanh()

        self.backbone.fc = torch.nn.Sequential(*layers)
        self.loss = torch.nn.SmoothL1Loss

    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        imgs, targets = batch
        preds = self.forward(imgs)
        # loss = F.smooth_l1_loss(preds, targets)
        loss = F.mse_loss(preds, targets)
        # preds = torch.reshape(preds.cpu(), (-1,))
        # targets = torch.reshape(targets.cpu(), (-1,))

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
        wandb.log(dict(
            loss=loss.item(),
            batch_nb=batch_idx,
            r2_calories=r2_calories,
            r2_proteins=r2_proteins,
            r2_fat=r2_fat,
            r2_sodium=r2_sodium,
            r2_overall=r2_overall,

        ))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
