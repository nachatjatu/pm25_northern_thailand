from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import lightning as L

import src.Data
import src.Transforms
import src.Models
import src.Utils

import torchvision.transforms as T
import os
import torch
                   
torch.set_printoptions(precision=2, sci_mode=False, linewidth=80)

def main(args):
    # set up Logger and Trainer
    exp_name = f"{args.model}_job{os.getenv('SLURM_JOB_ID', 'default')}"

    callbacks = src.Utils.init_callbacks(args)
    logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0
    )

    # set up data and transformations
    std_indices = [0, 6, 7, 8, 9, 10, 11, 13, 15, 16]
    norm_indices = [1, 2, 3, 4, 5, 12, 14]

    means, stds, mins, maxs = src.Utils.init_norm_std(args)
    print('Means:\n', means)
    print('SDs:\n', stds)
    print('Mins:\n', mins)
    print('Maxs:\n', maxs)

    normalize = src.Transforms.Normalize(mins, maxs, norm_indices)
    standardize = src.Transforms.Standardize(means, stds, std_indices)

    train_transform = T.Compose(
        [normalize, standardize, T.RandomCrop(256)]
    )

    val_transform = T.Compose(
        [normalize, standardize, T.Compose(
            [T.FiveCrop(256), T.Lambda(
                lambda crops: torch.stack([crop for crop in crops])
            )]
        )]
    )

    test_transform = val_transform

    # test norm, std
    """
    train_dataset = src.Data.PM25Dataset(path=os.path.join(args.root, 'train'), transform=T.Compose([normalize, standardize]))
    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    means, stds, mins, maxs = src.Utils.compute_dataset_statistics(dataloader)
    """

    pm25_data = src.Data.PM25DataModule(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        collate_fn=src.Utils.collate_fn
    )

    # set up model
    loss_fn = torch.nn.MSELoss()
    model = src.Utils.init_model(args, loss_fn, pm25_data)
        
    # train and validate model
    trainer.fit(model, pm25_data)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, default='UNet_v1')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0)

    args = parser.parse_args()

    main(args)