import os

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import src.PM25Transforms as PM25Transforms
import torchvision.transforms as transforms
from src.PM25Dataset import PM25Dataset
from src.PM25Stats import PM25Stats
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader
from src.PM25Transforms import RandomFlip, RandomRotate
from src.PM25UNet import PM25UNet, PM25ArgParser

def main():
    # argument parsing for SLURM
    print('Parsing arguments...')
    parser = PM25ArgParser()
    args = parser.parse_args()

    # set up folder paths
    print('Setting up folder paths...')
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')

    # set up transformations
    print('Computing band statistics...')
    mean, std = PM25Stats(train_path, args.batch_size, args.num_workers).compute_statistics()
    print(f'mean: {mean}')
    print(f'std: {std}')

    print('Setting up transformations...')
    to_tensor = PM25Transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    flip = RandomFlip()
    rotate = RandomRotate()

    # create Datasets, GeoSamplers, DataLoaders
    print('Initializing datasets, samplers, and dataloaders...')
    train_dataset = PM25Dataset(train_path, transforms=transforms.Compose([to_tensor, normalize, flip, rotate]))
    val_dataset = PM25Dataset(val_path, transforms=transforms.Compose([to_tensor, normalize]))

    train_sampler = PreChippedGeoSampler(train_dataset, shuffle=True)
    val_sampler = PreChippedGeoSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=args.batch_size, num_workers=args.num_workers)

    # set up model and Lightning trainer
    print('Initializing model and trainer...')
    model = PM25UNet(6, 1, args.lr)

    trainer = L.Trainer(max_epochs=args.epochs,
                        callbacks=[EarlyStopping(monitor="val_loss", 
                                                 mode="min",
                                                 patience=5,
                                                 divergence_threshold=1e9)]
    )

    # train and validate model
    print('Begin training...')
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders = val_dataloader
    )



if __name__ == '__main__':
    main()
