import optuna
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


from argparse import ArgumentParser
import lightning as L

import src.Data
import src.Transforms
import src.Models
import src.Utils

import torchvision.transforms as T
import os
import torch
                   
torch.set_printoptions(precision=2, sci_mode=False, linewidth=80)

def init_model(args, loss_fn, pm25_data, lr, weight_decay):
    pm25_data.setup()
    train_dataloader = pm25_data.train_dataloader()
    in_channels = next(iter(train_dataloader))[0].shape[1]

    model_class = getattr(src.Models, args.model)

    model = model_class(
        in_channels=in_channels, 
        out_channels=1, 
        lr=lr, 
        loss_fn=loss_fn,
        weight_decay=weight_decay,
        num_layers=args.num_layers,
        base_channels=args.base_channels
    )
    
    return model


def objective(trial, args):
    # Suggest values for learning rate and Huber delta
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    huber_delta = trial.suggest_uniform("huber_delta", 1.0, 10.0)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)


    # set up Logger and Trainer
    exp_name = (f"{args.model}/{os.getenv('SLURM_JOB_ID', 'default')}"
                f"/lr{lr}_wd{weight_decay}_bs{args.batch_size}"
                f"_nl{args.num_layers}_bc{args.base_channels}"
                f"/{os.getenv('SLURM_JOB_ID', 'default')}")

    # Add early stopping and pruning callback
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    ckpt_callback = ModelCheckpoint(
        dirpath='models/'+exp_name,
        filename="{epoch:02d}-{val_loss:.4f}",  
        monitor="val_loss",          
        save_top_k=3,                
        mode="min",                  
        save_last=True,              
        verbose=True
    )

    callbacks = [early_stop_callback, pruning_callback, ckpt_callback]
    logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1
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
        [normalize, standardize, T.RandomCrop(128)]
    )
    print(train_transform)

    val_transform = T.Compose(
        [normalize, standardize, T.Compose(
            [T.FiveCrop(128), T.Lambda(
                lambda crops: torch.stack([crop for crop in crops])
            )]
        )]
    )
    print(val_transform)

    test_transform = T.Compose(
        [normalize, standardize, T.Compose(
            [T.FiveCrop(128), T.Lambda(
                lambda crops: torch.stack([crop for crop in crops])
            )]
        )]
    )
    print(test_transform)

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
    loss_fn = torch.nn.HuberLoss(delta=huber_delta)
    model = init_model(args, loss_fn, pm25_data, lr, weight_decay)

    print(model)

    trainer.fit(model, pm25_data)

    return trainer.callback_metrics["val_loss"].item()


def main(args):
    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=20)

    print("Best hyperparameters:", study.best_params)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, default='UNet_v2')
    parser.add_argument("--num_workers", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--base_channels", type=int, default=64)

    args = parser.parse_args()

    main(args)