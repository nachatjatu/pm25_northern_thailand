import os

from src.PM25Stats import PM25Stats
import src.PM25Transforms as PM25Transforms
import torchvision.transforms as transforms
from src.PM25Dataset import PM25Dataset
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader
from src.PM25UNet import PM25ArgParser
import torch
import torch.nn as nn

def main():
    print('Parsing arguments...')
    parser = PM25ArgParser()
    args = parser.parse_args()

    print('Setting up folder paths...')
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')


    print('Computing band statistics...')
    mean, std, min, max = PM25Stats(train_path, args.batch_size, args.num_workers).compute_statistics()
    print(f'mean: {mean}')
    print(f'std: {std}')
    print(f'min: {min}')
    print(f'max: {max}')

    print('Setting up transformations...')
    to_tensor = PM25Transforms.ToTensor()
    normalize = PM25Transforms.Normalize(min, max, [4, 5])
    standardize = PM25Transforms.Standardize(mean, std, [0, 1, 2, 3, 6])
    transform = transforms.Compose([to_tensor, normalize, standardize])

    print('Testing normalization and standardization')
    mean, std, min, max = PM25Stats(train_path, args.batch_size, args.num_workers, transform).compute_statistics()
    print(f'mean: {mean}')
    print(f'std: {std}')
    print(f'min: {min}')
    print(f'max: {max}')


    print('Initializing datasets, samplers, and dataloaders...')
    val_dataset = PM25Dataset(val_path, transforms=transform)

    val_sampler = PreChippedGeoSampler(val_dataset, shuffle=False)

    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=args.batch_size, num_workers=args.num_workers)

    loss_list = []
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    # loss_fn = nn.MSELoss()
    with torch.no_grad():  # No need for gradients
        for _, y_tomorrow in val_dataloader:
            y_pred_persistence = torch.zeros_like(y_tomorrow)
            batch_loss = loss_fn(y_tomorrow, y_pred_persistence)
            loss_list.append(batch_loss)

    final_loss = sum(loss_list) / len(loss_list)
    print(f"Persistence Model Validation Loss: {final_loss:.4f}")

if __name__ == '__main__':
    main()
