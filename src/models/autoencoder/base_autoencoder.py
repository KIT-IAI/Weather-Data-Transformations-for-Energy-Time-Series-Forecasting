import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from xarray.core.utils import decode_numpy_dict_values


class BaseAutoencoder(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.1
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        result = torch.reshape(dec, (x.shape))
        return result

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def _common_step(self, batch, batch_idx, stage: str):
        loss = getattr(F, self.hparams.loss)(batch[0], self(batch[0]))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss
