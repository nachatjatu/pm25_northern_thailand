import os
import math

import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.compute_statistics import ComputeStatistics
import src.PM25Transforms as PM25Transforms
import torchvision.transforms as transforms
from src.PM25Dataset import PM25Dataset
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader

from src.PM25UNet import PM25UNet, PM25ArgParser
import matplotlib.pyplot as plt

# set up folder paths
cwd = os.getcwd()
folder = os.path.join(cwd, 'data/dataset_1')
train_path = os.path.join(folder, 'train')
val_path = os.path.join(folder, 'val')

model = PM25UNet.load_from_checkpoint(os.path.join(cwd, 'checkpoints/tmpf127_fzk'), in_channels=6, out_channels=1,map_location=torch.device('cpu'))

# set up transformations
mean, std = ComputeStatistics(train_path).compute_mean_std()
mean[4], std[4] = 0, 1
to_tensor = PM25Transforms.ToTensor()
normalize = transforms.Normalize(mean, std)
transform = transforms.Compose([to_tensor, normalize])

# create Datasets
val_dataset = PM25Dataset(val_path, transforms=transform)

# create GeoSamplers
val_sampler = PreChippedGeoSampler(val_dataset, shuffle = False)

# create DataLoaders
val_dataloader = DataLoader(
    val_dataset, sampler = val_sampler, batch_size = 1)

for input_bands, ground_truth in val_dataloader:
    predicted_image = model(input_bands)[0][0].detach().numpy()
    ground_truth_image = ground_truth[0][0].detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the images
    axes[0].imshow(ground_truth_image, cmap='viridis')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')  # Hide axis

    axes[1].imshow(predicted_image, cmap='viridis')
    axes[1].set_title("Predicted")
    axes[1].axis('off')  # Hide axis

    # Show the images side by side
    plt.show()