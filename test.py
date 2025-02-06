import os

import torch

from src.PM25Stats import PM25Stats
import Transforms as Transforms
import torchvision.transforms as transforms
from Dataset import PM25Dataset
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader

from Models import PM25UNet, PM25ArgParser
from SimpleConv import PM25SimpleConv
import matplotlib.pyplot as plt

# set up folder paths
cwd = os.getcwd()
folder = os.path.join(cwd, 'data/dataset_1')
train_path = os.path.join(folder, 'train')
val_path = os.path.join(folder, 'val')

# model = PM25UNet.load_from_checkpoint(
#     os.path.join(cwd, 'lightning_logs/version_59377634/checkpoints/epoch=39-step=11160.ckpt'), 
#     in_channels=6, out_channels=1,map_location=torch.device('cpu'))

# set up transformations
mean, std, min, max = PM25Stats(train_path, 1, 0).compute_statistics()
print('bands: PM2.5_tmr, u-wind_tdy, v-wind_tdy, aai_tdy, frp_tdy, elevation, delta_PM2.5')
print('mean:', mean)
print('std:', std)
print('min:', min)
print('max:', max)


to_tensor = Transforms.ToTensor()
normalize = Transforms.Normalize(min, max, [4, 5])
standardize = Transforms.Standardize(mean, std, [0, 1, 2, 3, 6])
transform = transforms.Compose([to_tensor, normalize, standardize])

mean, std, min, max = PM25Stats(train_path, 1, 0, transform).compute_statistics()
print('bands: PM2.5_tmr, u-wind_tdy, v-wind_tdy, aai_tdy, frp_tdy, elevation, delta_PM2.5')
print('mean:', mean)
print('std:', std)
print('min:', min)
print('max:', max)

# # create Datasets
val_dataset = PM25Dataset(val_path, transforms=transform)

# # create GeoSamplers
val_sampler = PreChippedGeoSampler(val_dataset, shuffle = False)

# create DataLoaders
val_dataloader = DataLoader(
    val_dataset, sampler = val_sampler, batch_size = 1)

for input_bands, ground_truth in val_dataloader:
    predicted_image = input_bands[0][0].detach().numpy()
    # predicted_image = model(input_bands)[0][0].detach().numpy()

    ground_truth_image = ground_truth[0][0].detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the images
    im1 = axes[0].imshow(ground_truth_image, cmap='viridis')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')  # Hide axis
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical')

    im2 = axes[1].imshow(predicted_image, cmap='viridis')
    axes[1].set_title("Predicted")
    axes[1].axis('off')  # Hide axis
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical')

    # Show the images side by side
    plt.show()