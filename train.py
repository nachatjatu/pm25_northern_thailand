from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L

import src.Data
import src.Transforms
import src.Models

import torchvision.transforms as T
import os
import torch

def init_callbacks(args):
    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,  
        mode="min",  
        verbose=True
    )
    ckpt_callback = ModelCheckpoint(
        dirpath=f"models/{args.model}/lr{args.lr}_bs{args.batch_size}/{os.getenv('SLURM_JOB_ID', 'default')}",          
        filename="{epoch:02d}-{val_loss:.4f}",  
        monitor="val_loss",          
        save_top_k=3,                
        mode="min",                  
        save_last=True,              
        verbose=True
    )
    return [es_callback, ckpt_callback]

def init_norm_std(args):
    train_dataset = src.Data.PM25Dataset(path=os.path.join(args.root, 'train'))
    return train_dataset.compute_dataset_statistics(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

def collate_fn(batch):
    """
    Flattens the batch when FiveCrop is used, so each crop is treated as an independent sample.
    """
    inputs, outputs = zip(*batch)  
    inputs = torch.cat(inputs, dim=0)  
    outputs = torch.cat(outputs, dim=0) 
    
    return inputs, outputs

def main(args):
    callbacks = init_callbacks(args)
    exp_name = f"{args.model}_job{os.getenv('SLURM_JOB_ID', 'default')}"

    logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    trainer = L.Trainer(
        max_steps=args.max_steps,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=2000
    )

    band_indices = None
    std_indices = [0, 6, 7, 8, 9, 10, 11, 13, 15, 16]
    norm_indices = [1, 2, 3, 4, 5, 12, 14]

    mean, sd, min_val, max_val = init_norm_std(args)

    normalize = src.Transforms.Normalize(min_val, max_val, norm_indices)
    standardize = src.Transforms.Standardize(mean, sd, std_indices)
    # flip = src.Transforms.RandomFlip(6, 7)
    # rotate = src.Transforms.RandomRotate(6, 7)

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

    test_transform = T.Compose(
        [normalize, standardize, T.Compose(
            [T.FiveCrop(256), T.Lambda(
                lambda crops: torch.stack([crop for crop in crops])
            )]
        )]
    )

    pm25_data = src.Data.PM25DataModule(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        band_indices=band_indices,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        collate_fn=collate_fn
    )

    pm25_data.setup()
    train_dataloader = pm25_data.train_dataloader()
    in_channels = next(iter(train_dataloader))[0].shape[1]

    loss_fn = torch.nn.MSELoss()

    if args.model == 'Persistence':
        model = src.Models.Persistence(out_channels=1, loss_fn=loss_fn)
        results = trainer.validate(model, datamodule=pm25_data)
        print(results)
        return
    
    if args.model == 'UNet_v1':
        model = src.Models.UNet_v1(
            in_channels=in_channels, 
            out_channels=1, 
            lr=args.lr, 
            loss_fn=loss_fn
        )

    elif args.model == 'SimpleConv':
        model = src.Models.SimpleConv(
            in_channels=in_channels, 
            out_channels=1, 
            lr=args.lr, 
            loss_fn=loss_fn
        )
        
    # train and validate model
    trainer.fit(model, pm25_data)




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, default='UNet_v1')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50000)

    args = parser.parse_args()

    main(args)