import os

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import src.PM25Transforms as PM25Transforms
import torchvision.transforms as transforms
from src.PM25Dataset import PM25Dataset
from src.PM25Stats import PM25Stats
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader

from src.PM25UNet import PM25UNet, PM25ArgParser
from lightning.pytorch.loggers import TensorBoardLogger


def main():
    # argument parsing for SLURM
    parser = PM25ArgParser()
    args = parser.parse_args()

    # set up folder paths
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')

    # set up transformations
    to_tensor = PM25Transforms.ToTensor()
    mean, std = PM25Stats(train_path, args.batch_size, args.num_workers).compute_statistics()

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

    trainer.save_checkpoint(args.save_path)

    return 0

if __name__ == '__main__':
    main()