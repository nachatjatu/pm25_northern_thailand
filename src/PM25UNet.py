import torch
from torch import nn, optim
import lightning as L
import argparse
from torch.optim.lr_scheduler import LinearLR


class PM25ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PM2.5 UNet ArgParser")
        self.parser.add_argument('--epochs', type=int, default=50, 
                                 help='Number of training epochs')
        self.parser.add_argument("--num_workers", type=int, default=0, 
                                 help="Number of workers")
        self.parser.add_argument("--batch_size", type=int, default=8, 
                                 help="Batch size")
        self.parser.add_argument("--lr", type=float, default=0.0001, 
                                 help="Learning rate")
        self.parser.add_argument("--data_path", type=str, default="./data/dataset_1", 
                                 help="Path to dataset")

    def parse_args(self):
        return self.parser.parse_args()
    

class DownBlock(nn.Module):
    """A DownBlock module implementation for a U-Net

    DownBlock implements one U-Net downsampling block, consisting of two
    convolutions (w/ same padding) followed by a max pool. 

    Attributes:
        in_channels (int):  # of input channels
        out_channels (int): # of output channels 
    
    Methods:
        forward(x): passes input through DownBlock, returns output tensor
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        """Initializes DownBlock

        Args:
            in_channels (int):  # of input channels
            out_channels (int): # of output channels
        """
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout)
        )
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        """Performs forward pass on an input

        Args:
            x (torch.Tensor):   input tensor of shape 
                                (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor, torch.Tensor: output tensors after convolution and
                                        max pooling, respectively
        """
        x_conv = self.conv(x)
        x_down = self.pool(x_conv)
        return x_conv, x_down # use x_conv for skip connections
    

class BottleneckBlock(nn.Module):
    """A BottleneckBlock implementation for a U-Net

    BottleneckBlock implements one U-Net bottleneck block, consisting of two
    convolutions (w/ same padding). 

    Attributes:
        in_channels (int):  # of input channels 
        out_channels (int): # of output channels 
    
    Methods:
        forward(x): passes input through BottleneckBlock, returns output tensor
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        """Initializes BottleneckBlock

        Args:
            in_channels (int):  # of input channels
            out_channels (int): # of output channels
        """
        super(BottleneckBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x):
        """Performs forward pass on an input x

        Args:
            x (torch.tensor):   input tensor of shape 
                                (batch_size, in_channels, height, width)

        Returns:
            torch.tensor: output tensor after convolutions
        """
        return self.conv(x)


class UpBlock(nn.Module):
    """A UpBlock module implementation for a U-Net

    UpBlock implements one U-Net upsampling block, consisting of a 
    deconvolution followed by concatenation with a skip connection and
    two convolutions (w/ same padding).

    Attributes:
        in_channels (int):      # of input channels
        out_channels (int):     # of output channels 
        skip (torch.tensor):    Skip connection tensor
    
    Methods:
        forward(x, skip): Passes inputs through UpBlock, returns output tensor
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        """Initializes UpBlock

        Args:
            in_channels (int):  # of input channels
            out_channels (int): # of output channels
        """
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                     kernel_size = 2, stride = 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x, skip):
        """Performs forward pass on an input and skip connection 

        Args:
            x (torch.tensor):       Input tensor of shape 
                                    (batch_size, in_channels, height, width)
            skip (torch.tensor):    Skip connection tensor of shape
                                    (batch_size, in_channels, height, width)

        Returns:
            torch.tensor:   Output tensor after deconvolution, concatenation,
                            and convolutions
        """
        x = self.up(x)
        x = torch.cat((x, skip), dim = 1)
        x = self.conv(x)
        return x
    

class PM25UNet(L.LightningModule):
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
    def __init__(self, in_channels, out_channels, lr=1e-4, loss_fn=None):
        super(PM25UNet, self).__init__()
        self.lr = lr
        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()

        print(f'Using learning rate {self.lr}')
        print(f'Using loss_fn {self.loss_fn}')

        self.down1 = DownBlock(in_channels, 64) 
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)       
        self.down4 = DownBlock(256, 512) 
        self.bottleneck = BottleneckBlock(512, 1024)
        self.up4 = UpBlock(1024, 512)
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward(self, x):
        # pass image through downsampling blocks, retaining skip connections
        x1_conv, x1_down = self.down1(x)
        x2_conv, x2_down = self.down2(x1_down)
        x3_conv, x3_down = self.down3(x2_down)
        x4_conv, x4_down = self.down4(x3_down)

        # pass image through bottleneck block
        x_bottleneck = self.bottleneck(x4_down)

        # pass image through upsampling blocks, maintaining skip connections
        x4_up = self.up4(x_bottleneck, x4_conv)
        x3_up = self.up3(x4_up, x3_conv)
        x2_up = self.up2(x3_up, x2_conv)
        x1_up = self.up1(x2_up, x1_conv)

        # pass image through output layer and return
        return self.out(x1_up)

    def training_step(self, batch, _):
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
    
