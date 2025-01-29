# Import required packages
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.PM25Transforms import RandomFlip, RandomRotate, RandomCompose, ToTensor
from src.PM25Dataset import PM25Dataset
from src.PM25UNet import PM25UNet
from src.compute_statistics import ComputeStatistics
from torchvision import transforms
from torchgeo.samplers import PreChippedGeoSampler
from tqdm import tqdm
import argparse

# argument parsing for SLURM
parser = argparse.ArgumentParser(description="Train U-Net on SLURM")
parser.add_argument("--epochs", type=int, default=50, 
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, 
                    help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, 
                    help="Learning rate")
parser.add_argument("--data_path", type=str, default="./data", 
                    help="Path to dataset")
parser.add_argument("--save_path", type=str, default="./checkpoints", 
                    help="Path to save model")
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up folder paths
train_path = os.path.join(args.data_path, 'train')
val_path = os.path.join(args.data_path, 'val')
test_path = os.path.join(args.data_path, 'test')

# precompute mean and std for standardization
mean, std = ComputeStatistics(train_path).compute_mean_std()
# exempt fire band from standardization (highly skewed)
mean[4] = 0
std[4] = 1

# set up transformations
transform = transforms.Compose([ToTensor(), transforms.Normalize(mean, std)])

# create Datasets
train_dataset = PM25Dataset(train_path, transforms=transform)
val_dataset = PM25Dataset(val_path, transforms=transform)
test_dataset = PM25Dataset(test_path, transforms=transform)

# create GeoSamplers
train_sampler = PreChippedGeoSampler(train_dataset, shuffle = True)
val_sampler = PreChippedGeoSampler(val_dataset, shuffle = False)
test_sampler = PreChippedGeoSampler(test_dataset, shuffle = False)

# create DataLoaders
train_dataloader = DataLoader(
    train_dataset, sampler = train_sampler, batch_size = args.batch_size)
val_dataloader = DataLoader(
    val_dataset, sampler = val_sampler, batch_size = args.batch_size)
test_dataloader = DataLoader(
    test_dataset, sampler = test_sampler, batch_size = args.batch_size)

print('Training set has {} instances'.format(len(train_dataset)))
print('Validation set has {} instances'.format(len(val_dataset)))
print('Test set has {} instances'.format(len(test_dataset)))

# set up model
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # He Initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model = PM25UNet(in_channels = 6, out_channels = 1)
model.apply(init_weights)

# set up loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

def train():
    # training loop
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        train_running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                            position = 0, leave = True)
        for idx, (input_bands, true_pm25) in enumerate(progress_bar):
            if torch.any(torch.isnan(input_bands)):
                print("NaN input detected!")
            if torch.any(torch.isnan(true_pm25)):
                print("NaN target detected!")
            
            optimizer.zero_grad()

            pred_pm25 = model(input_bands)

            true_nan_mask = torch.isnan(true_pm25)
            valid_mask = ~true_nan_mask

            loss = loss_fn(pred_pm25[valid_mask], true_pm25[valid_mask])
                
            loss.backward()

            optimizer.step()

            train_running_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss = train_running_loss / (idx + 1)
        train_losses.append(train_loss)
        
        # validate
        model.eval()
        val_running_loss = 0.0   
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                                position = 0, leave = True)
            for idx, (input_bands, true_pm25) in enumerate(progress_bar): 
                if torch.any(torch.isnan(input_bands)):
                    print("NaN input detected!")
                if torch.any(torch.isnan(true_pm25)):
                    print("NaN target detected!")

                pred_pm25 = model(input_bands)

                true_nan_mask = torch.isnan(true_pm25)
                valid_mask = ~true_nan_mask

                loss = loss_fn(pred_pm25[valid_mask], true_pm25[valid_mask])

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)
            val_losses.append(val_loss)
        
        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), args.save_path)
        
if __name__ == "__main__":
    train()