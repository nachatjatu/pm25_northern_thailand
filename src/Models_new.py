
import torch
from torch import nn, optim
import lightning as L
import torchvision.transforms.functional as F
import segmentation_models_pytorch as smp

class UNet_v5(L.LightningModule):
    def __init__(self, in_channels, lr=1e-4, weight_decay=1e-5, unfreeze_epoch=5):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",  
            encoder_weights="imagenet",  
            in_channels=in_channels,  
            classes=1,  
            activation=None,
            encoder_depth=2,
            decoder_channels=(64, 32)
        )

        self.lr = lr
        self.weight_decay = weight_decay

        self.criterion = nn.L1Loss()
        self.unfreeze_epoch = unfreeze_epoch  # Epoch at which to start unfreezing
        self.current_unfreeze_step = 0  # Track the number of layers unfrozen

        # Initially, freeze the encoder (ResNet34 backbone)
        self.freeze_encoder()
    
    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        layers = [
            self.model.encoder.layer3,
            self.model.encoder.layer2,
            self.model.encoder.layer1,  # Shallowest
        ]

        if self.current_unfreeze_step < len(layers):
            for param in layers[self.current_unfreeze_step].parameters():
                param.requires_grad = True
            print(f"Unfreezing Layer {self.current_unfreeze_step + 1}")
            self.current_unfreeze_step += 1  # Move to the next layer

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat  = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat  = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def on_train_epoch_start(self):
        """Gradually unfreeze the encoder starting from the deepest layers."""
        if self.current_epoch >= self.unfreeze_epoch:
            self.unfreeze_encoder()  # Unfreeze one layer at a time

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss_epoch")
        val_loss = self.trainer.callback_metrics.get("val_loss")

        if train_loss is not None and val_loss is not None:
            self.logger.experiment.add_scalars(
                "Losses",
                {"Train Loss": train_loss, "Validation Loss": val_loss},
                self.current_epoch
            )