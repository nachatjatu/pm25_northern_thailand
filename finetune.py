from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L

import src.Data
import src.Transforms
import src.Models_new as M
import src.Utils

import torchvision.transforms as T
import os
import torch
                   
torch.set_printoptions(precision=2, sci_mode=False, linewidth=80)

def main(args):
    # set up Logger and Trainer
    exp_name = (f"{os.getenv('SLURM_JOB_ID', 'default')}"
                f"/lr{args.lr}_bs{args.batch_size}"
                f"/{os.getenv('SLURM_JOB_ID', 'default')}")

    callbacks = src.Utils.init_callbacks(args, exp_name)
    logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1,
    )

    # set up data and transformations
    std_indices = [0, 1, 2, 3, 4, 5, 8, 9]
    norm_indices = [6, 7]

    means, stds, mins, maxs = src.Utils.init_norm_std(args)
    print('Means:\n', means)
    print('SDs:\n', stds)
    print('Mins:\n', mins)
    print('Maxs:\n', maxs)

    normalize = src.Transforms.Normalize(mins, maxs, norm_indices)
    standardize = src.Transforms.Standardize(means, stds, std_indices)

    train_transform = T.Compose(
        [normalize, standardize]
    )
    print(train_transform)

    val_transform = train_transform
    print(val_transform)

    test_transform = train_transform
    print(test_transform)

    pm25_data = src.Data.PM25DataModule(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )

    # set up model
    model = M.UNet_v5(in_channels=9, lr=args.lr, weight_decay=args.weight_decay)

    print(model)

    trainer.fit(model, pm25_data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0)

    args = parser.parse_args()

    main(args)