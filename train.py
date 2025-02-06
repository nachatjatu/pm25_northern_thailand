import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import src.Dataset as Dataset

import src.Models as Models
from argparse import ArgumentParser

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
        dirpath=f"models/{args.model_name}/lr{args.lr}_bs{args.batch_size}/",          
        filename="{epoch:02d}-{val_loss:.4f}",  
        monitor="val_loss",          
        save_top_k=3,                
        mode="min",                  
        save_last=True,              
        verbose=True
    )
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # set up DataModule and model
    band_indices = None
    std_indices = [0, 1, 2, 3, 6]
    norm_indices = [4, 5]
    pm25 = Dataset.PM25DataModule('./data/dataset_2', args.batch_size, args.num_workers, norm_indices, std_indices, band_indices=band_indices)

    pm25.setup(stage="fit")
    train_dataloader = pm25.train_dataloader()
    in_channels = next(iter(train_dataloader))[0].shape[1]

    if args.model_name == 'Persistence':
        model = Models.Persistence(out_channels=1)
        results = trainer.validate(model, datamodule=pm25)
        print(results)
        return
    
    if args.model_name == 'UNet_v1':
        model = Models.UNet_v1(**dict_args, in_channels=in_channels, out_channels=1)
    elif args.model_name == 'SimpleConv':
        model = Models.SimpleConv(**dict_args, in_channels=in_channels, out_channels=1)
        

    # train and validate model
    trainer.fit(model, pm25)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default='UNet_v1')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=1)

    args = parser.parse_args()

    # train
    main(args)
