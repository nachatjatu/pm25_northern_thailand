import os
import math

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.compute_statistics import ComputeStatistics
import src.PM25Transforms as PM25Transforms
import torchvision.transforms as transforms
from src.PM25Dataset import PM25Dataset
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader

from src.PM25UNet import PM25UNet, PM25ArgParser
from lightning.pytorch.loggers import TensorBoardLogger


if __name__ == '__main__':
    # argument parsing for SLURM
    parser = PM25ArgParser()
    args = parser.parse_args()

    # set up folder paths
    cwd = os.getcwd()
    folder = os.path.join(cwd, 'data/dataset_1')
    train_path = os.path.join(folder, 'train')
    val_path = os.path.join(folder, 'val')
    test_path = os.path.join(folder, 'test')

    # set up transformations
    mean, std = ComputeStatistics(train_path).compute_mean_std()
    mean[4], std[4] = 0, 1
    to_tensor = PM25Transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([to_tensor, normalize])

    # create Datasets
    train_dataset = PM25Dataset(train_path, transforms=transform)
    val_dataset = PM25Dataset(val_path, transforms=transform)

    # create GeoSamplers
    train_sampler = PreChippedGeoSampler(train_dataset, shuffle = True)
    val_sampler = PreChippedGeoSampler(val_dataset, shuffle = False)

    # create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, sampler = train_sampler, batch_size = args.batch_size)
    val_dataloader = DataLoader(
        val_dataset, sampler = val_sampler, batch_size = args.batch_size)

    # set up model and Lightning trainer
    def kaiming_init(model):
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            elif name.startswith("layers.0"):  # The first layer does not have ReLU applied on its input
                param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
            else:
                param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))
    model = PM25UNet(6, 1)
    # kaiming_init(model)

    log_dir = '$SCRATCH/logs'
    logger = TensorBoardLogger(log_dir)
    
    trainer = L.Trainer(max_epochs=args.epochs, 
                        logger=logger,
                        callbacks=[EarlyStopping(monitor="val_loss", 
                                                 mode="min",
                                                 divergence_threshold=1e9),]
    )
    
    # train and validate model
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders = val_dataloader
    )

    trainer.save_checkpoint(os.path.join(cwd, 'checkpoints'))