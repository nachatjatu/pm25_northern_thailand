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
import torch
from src.PM25UNet import PM25UNet, PM25ArgParser
from lightning.pytorch.loggers import TensorBoardLogger
import tqdm


def main():
    # argument parsing for SLURM
    parser = PM25ArgParser()
    args = parser.parse_args()

    # set up folder paths
    cwd = os.getcwd()
    folder = os.path.join(cwd, 'data/dataset_1')
    train_path = os.path.join(folder, 'train')
    val_path = os.path.join(folder, 'val')

    # set up transformations
    to_tensor = PM25Transforms.ToTensor()

    train_dataset = PM25Dataset(train_path, transforms=to_tensor)
    train_sampler = PreChippedGeoSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=args.batch_size, num_workers=args.num_workers)
    
    for input_bands, ground_truth in train_dataloader:
        num_bands = input_bands.shape[1] + ground_truth.shape[1]
        image_height, image_width = input_bands.shape[2], input_bands.shape[3]
        break
    
    psum = torch.zeros(num_bands)
    psum_sq = torch.zeros(num_bands)

    for input_bands, ground_truth in train_dataloader:
        image = torch.cat((input_bands, ground_truth), dim = 1)
        psum += image.sum(axis = [0, 2, 3])
        psum_sq += (image ** 2).sum(axis = [0, 2, 3])
    
    count = len(train_dataset) * image_height * image_width
    mean = psum / count
    var = (psum_sq / count) - (mean) ** 2
    std = torch.sqrt(var)

    print(f'mean: {mean}')
    print(f'std: {std}')

    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([to_tensor, normalize])

    # create Datasets
    train_dataset = PM25Dataset(train_path, transforms=transform)
    val_dataset = PM25Dataset(val_path, transforms=transform)

    # create GeoSamplers
    train_sampler = PreChippedGeoSampler(train_dataset, shuffle=True)
    val_sampler = PreChippedGeoSampler(val_dataset, shuffle=False)

    # create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=args.batch_size, num_workers=args.num_workers)

    # set up model and Lightning trainer
    model = PM25UNet(6, 1)

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

    return 0

if __name__ == '__main__':
    main()