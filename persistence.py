import os


import src.PM25Transforms as PM25Transforms
import torchvision.transforms as transforms
from src.PM25Dataset import PM25Dataset
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader
from src.PM25UNet import PM25ArgParser
import torch
from sklearn.metrics import mean_squared_error

def main():
    print('Parsing arguments...')
    parser = PM25ArgParser()
    args = parser.parse_args()

    print('Setting up folder paths...')
    val_path = os.path.join(args.data_path, 'val')


    print('Setting up transformations...')
    to_tensor = PM25Transforms.ToTensor()
    transform = transforms.Compose([to_tensor])

    print('Initializing datasets, samplers, and dataloaders...')
    val_dataset = PM25Dataset(val_path, transforms=transform)

    val_sampler = PreChippedGeoSampler(val_dataset, shuffle=False)

    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=args.batch_size, num_workers=args.num_workers)

    mse_list = []

    with torch.no_grad():  # No need for gradients
        for _, y_tomorrow in val_dataloader:
            y_pred_persistence = torch.zeros_like(y_tomorrow)
            batch_mse = mean_squared_error(y_tomorrow.numpy().flatten(), y_pred_persistence.numpy().flatten())
            mse_list.append(batch_mse)

    final_mse = sum(mse_list) / len(mse_list)
    print(f"Persistence Model Validation MSE: {final_mse:.4f}")

if __name__ == '__main__':
    main()
