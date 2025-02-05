from torch import nn, optim
import lightning as L

class PM25SimpleConv(L.LightningModule):
    """
    A U-Net implementation for day-to-day PM2.5 image prediction using
    PyTorch Lightning.

    This class defines a U-Net architecture with an encoder-decoder structure 
    and skip connections.

    Attributes:
        in_channels (int): # of input channels (# of bands in multi-band image)
        out_channels (int): # of output channels (usually 1)

    Methods:
        forward(x): Passes input tensor through U-Net, returns output tensor
        training_step(self, batch, _): Performs one step in the training loop
        validation_step(self, batch, _): Performs one step in the val loop
        test_step(self, batch, _): Performs one step in the testing loop
    """
    def __init__(self, in_channels, out_channels, lr=1e-5, loss_fn=None):
        super(PM25SimpleConv, self).__init__()
        self.lr = lr
        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x):
        return self.conv(x)

    def training_step(self, batch, _):
        if self.trainer.is_last_batch:
            print(self.trainer.optimizers[0].param_groups[0]['lr'])
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("val_loss", loss, on_epoch=True)
    
    def test_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("test_loss", loss)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Step the scheduler every epoch
                "frequency": 1,  # Apply it every epoch
                "monitor": "val_loss"
            },
        }
    
