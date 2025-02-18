import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import src.Data as Data
import src.Models as Models
import src.Transforms as Transforms
from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger
import os
import torchvision.transforms as transforms
import torch.nn as nn

def main(args):
    dict_args = vars(args)

    # set up Lightning Trainer and callbacks
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,  
        mode="min",  
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/{args.model_name}/lr{args.lr}_bs{args.batch_size}_loss_fn{args.loss_fn}/{os.getenv('SLURM_JOB_ID', 'default')}",          
        filename="{epoch:02d}-{val_loss:.4f}",  
        monitor="val_loss",          
        save_top_k=3,                
        mode="min",                  
        save_last=True,              
        verbose=True
    )
    experiment_name = f"{args.model_name}_job{os.getenv('SLURM_JOB_ID', 'default')}"
    logger = TensorBoardLogger(save_dir="logs", name=experiment_name)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # set up DataModule   
    band_indices = None
    std_indices = [0, 8, 9, 10, 11, 12, 13, 17, 18]
    norm_indices = [1, 2, 3, 4, 5, 6, 7, 14, 15, 16]

    transforms_dict = {
        'train': None,
        'val': None,
        'test': None
    }

    train_dataset = Data.PM25Dataset(data_dir = args.data_path + '/train')
    mean, sd, min, max = train_dataset.compute_statistics(
        batch_size = args.batch_size
    )

    normalize = Transforms.Normalize(min, max, norm_indices)
    standardize = Transforms.Standardize(mean, sd, std_indices)
    flip = Transforms.RandomFlip(8, 9)
    rotate = Transforms.RandomRotate(8, 9)

    transforms_dict = {
        'train': transforms.Compose([normalize, standardize]),
        'val': transforms.Compose([normalize, standardize]),
        'test': transforms.Compose([normalize, standardize])
    }

    pm25_data = Data.PM25DataModule(
        args.data_path, transforms_dict, args.batch_size, args.num_workers, 
        select_bands=band_indices
    )

    pm25_data.setup()
    train_dataloader = pm25_data.train_dataloader()
    in_channels = next(iter(train_dataloader))[0].shape[1]

    # set up model
    loss_fn = nn.MSELoss() if args.loss_fn == 'mse' else nn.HuberLoss()

    if args.model_name == 'Persistence':
        model = Models.Persistence(out_channels=1, loss_fn=loss_fn)
        results = trainer.validate(model, datamodule=pm25_data)
        print(results)
        return
    if args.model_name == 'UNet_v1':
        model = Models.UNet_v1(in_channels=in_channels, out_channels=1, lr=args.lr, loss_fn=loss_fn)
    elif args.model_name == 'SimpleConv':
        model = Models.SimpleConv(in_channels=in_channels, out_channels=1, lr=args.lr, loss_fn=loss_fn)
        
    # train and validate model
    trainer.fit(model, pm25_data)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default='UNet_v1')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--loss_fn", type=str, default='mse')

    args = parser.parse_args()

    # train
    main(args)
