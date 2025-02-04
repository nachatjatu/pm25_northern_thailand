import os


import src.PM25Transforms as PM25Transforms
import torchvision.transforms as transforms
from src.PM25Dataset import PM25Dataset
from src.PM25Stats import PM25Stats
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader
from src.PM25UNet import PM25ArgParser
import torch
from sklearn.metrics import mean_squared_error

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
    # print('Computing band statistics...')
    # mean, std = PM25Stats(train_path, args.batch_size, args.num_workers).compute_statistics()
    # print(f'mean: {mean}')
    # print(f'std: {std}')

    print('Setting up transformations...')
    to_tensor = PM25Transforms.ToTensor()
    # normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([to_tensor])

    # create Datasets, GeoSamplers, DataLoaders
    print('Initializing datasets, samplers, and dataloaders...')
    val_dataset = PM25Dataset(val_path, transforms=transform)

    val_sampler = PreChippedGeoSampler(val_dataset, shuffle=False)

    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=args.batch_size, num_workers=args.num_workers)

    mse_list = []

    with torch.no_grad():  # No need for gradients
        for batch in val_dataloader:
            x_today, y_tomorrow = batch  # Unpack batch

            # Persistence Model: Predict tomorrow's PM2.5 as today's PM2.5 (first band)
            y_pred_persistence = x_today[:, 0, :, :]  # Extract first band

            # Compute MSE for this batch
            batch_mse = mean_squared_error(y_tomorrow.numpy().flatten(), y_pred_persistence.numpy().flatten())
            mse_list.append(batch_mse)

    # Compute final MSE
    final_mse = sum(mse_list) / len(mse_list)
    print(f"Persistence Model Validation MSE: {final_mse:.4f}")


if __name__ == '__main__':
    main()
