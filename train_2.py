from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger
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
    exp_name = (f"{args.model}/{os.getenv('SLURM_JOB_ID', 'default')}"
                f"/lr{args.lr}_bs{args.batch_size}_nl{args.num_layers}_bc{args.base_channels}"
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
    band_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    std_indices = [0, 1, 2, 8, 9]
    norm_indices = [3, 4, 5, 6, 7]

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
        collate_fn=src.Utils.collate_fn,
        select_indices=band_indices
    )

    # set up model
    loss_fn = torch.nn.GaussianNLLLoss()

    model = src.Utils.init_model(args, loss_fn, pm25_data, 2)

    print(model)

    if args.model == 'Persistence' or args.mode == 'val':
        trainer.validate(model, pm25_data)
    else:
        trainer.fit(model, pm25_data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument('--model', type=str, default='UNet_v3')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--delta", type=float, default=1.0)

    args = parser.parse_args()

    main(args)