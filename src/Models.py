
import torch
from torch import nn, optim
import lightning as L


class Persistence(L.LightningModule):
    def __init__(self, in_channels, out_channels, loss_fn, lr, weight_decay, num_layers, base_channels):
        super(Persistence, self).__init__()
        self.out_channels = out_channels
        self.loss_fn = loss_fn

    def forward(self, x):
        return torch.zeros(x.shape[0], self.out_channels, x.shape[2], x.shape[3], device=x.device)
    
    def validation_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("val_loss", loss, on_epoch=True)


class SimpleConv_v1(L.LightningModule):
    def __init__(self, in_channels, out_channels, loss_fn, lr, weight_decay, num_layers, base_channels):
        super(SimpleConv_v1, self).__init__()
        self.lr = lr
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay

        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x):
        return self.conv(x)

    def training_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("val_loss", loss, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
    

class SimpleConv_v2(L.LightningModule):
    def __init__(self, in_channels, out_channels, loss_fn, lr, weight_decay, num_layers, base_channels):
        super(SimpleConv_v2, self).__init__()
        self.lr = lr
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay

        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, bias=True, padding='same')

    def forward(self, x):
        return self.conv(x)

    def training_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("val_loss", loss, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
    

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
    def __init__(self, in_channels, out_channels):
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
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size = 3, padding = 'same')
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
    def __init__(self, in_channels, out_channels):
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
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.ReLU(inplace = True)
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
    def __init__(self, in_channels, out_channels):
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
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.ReLU(inplace = True),
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
    

class UNet_v1(L.LightningModule):
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
    def __init__(self, in_channels, out_channels, lr, loss_fn, weight_decay, num_layers, base_channels):
        super(UNet_v1, self).__init__()
        self.lr = lr
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay

        self.down1 = DownBlock(in_channels, 64) 
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)       
        self.bottleneck = BottleneckBlock(256, 512)
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size = 1)

    # def on_after_backward(self):
    #     """Called after loss.backward(), before optimizer step."""
    #     print("Gradient Magnitudes:")
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f"{name}: {param.grad.norm().item():.4f}")

    def forward(self, x):
        # pass image through downsampling blocks, retaining skip connections
        x1_conv, x1_down = self.down1(x)
        x2_conv, x2_down = self.down2(x1_down)
        x3_conv, x3_down = self.down3(x2_down)

        # pass image through bottleneck block
        x_bottleneck = self.bottleneck(x3_down)

        # pass image through upsampling blocks, maintaining skip connections
        x3_up = self.up3(x_bottleneck, x3_conv)
        x2_up = self.up2(x3_up, x2_conv)
        x1_up = self.up1(x2_up, x1_conv)

        # pass image through output layer and return
        return self.out(x1_up)

    def training_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.log("val_loss", loss, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer
    



class DownBlock_v2(nn.Module):
    """A DownBlock module implementation for a U-Net

    DownBlock implements one U-Net downsampling block, consisting of two
    convolutions (w/ same padding) followed by a max pool. 

    Attributes:
        in_channels (int):  # of input channels
        out_channels (int): # of output channels 
    
    Methods:
        forward(x): passes input through DownBlock, returns output tensor
    """
    def __init__(self, in_channels, out_channels):
        """Initializes DownBlock

        Args:
            in_channels (int):  # of input channels
            out_channels (int): # of output channels
        """
        super(DownBlock_v2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3, padding=0, bias=False),
            nn.ReflectionPad2d(padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, padding=0, bias=False),
            nn.ReflectionPad2d(padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)

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


class UpBlock_v2(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        """Initializes UpBlock

        Args:
            in_channels (int):  # of input channels
            out_channels (int): # of output channels
        """
        super(UpBlock_v2, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 
                                     kernel_size = 2, stride = 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 
                      kernel_size=3, padding=0, bias=False),
            nn.ReflectionPad2d(padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, padding=0, bias=False),
            nn.ReflectionPad2d(padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
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
    

class UNet_v2(L.LightningModule):
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
    def __init__(self, in_channels, out_channels, lr, loss_fn, 
                 weight_decay, base_channels=64, num_layers=3):
        super(UNet_v2, self).__init__()
        self.lr = lr
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.num_layers = num_layers

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        down_channels = [in_channels] + [base_channels * (2 ** i) for i in range(num_layers)]

        for i in range(num_layers):
            self.down_blocks.append(DownBlock_v2(down_channels[i], down_channels[i+1]))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(down_channels[-1], down_channels[-1] * 2, 
                      kernel_size = 3, padding = 0),
            nn.ReflectionPad2d(padding=1),
            nn.BatchNorm2d(down_channels[-1] * 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(down_channels[-1] * 2, down_channels[-1] * 2, 
                      kernel_size = 3, padding = 0),
            nn.ReflectionPad2d(padding=1),
            nn.BatchNorm2d(down_channels[-1] * 2),
            nn.ReLU(inplace = True)
        )

        up_channels = list(reversed(down_channels[1:]))
        
        for i in range(num_layers):
            self.up_blocks.append(UpBlock_v2(up_channels[i] * 2, up_channels[i]))

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size = 1)
    

    def forward(self, x):
        skip_connections = []

        for down_block in self.down_blocks:
            x_conv, x = down_block(x)
            skip_connections.append(x_conv)

        x = self.bottleneck(x)

        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x, skip)

        return self.out(x)
    

    def training_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, _):
        input_bands, true_pm25 = batch
        pred_pm25 = self(input_bands)
        loss = self.loss_fn(pred_pm25, true_pm25)
        self.logger.experiment.add_scalar("val_loss", loss, self.global_step)
        self.log("val_loss", loss, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer
    
