from src.PM25Transforms import ToTensor
from src.PM25Dataset import PM25Dataset
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader
import torch

class PM25Stats:
    def __init__(self, train_path, batch_size, num_workers, transform=ToTensor()):
        self.train_path = train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def compute_statistics(self):
        train_dataset = PM25Dataset(self.train_path, transforms=self.transform)
        train_sampler = PreChippedGeoSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=self.batch_size, num_workers=self.num_workers)
        
        for input_bands, ground_truth in train_dataloader:
            num_bands = input_bands.shape[1] + ground_truth.shape[1]
            image_height, image_width = input_bands.shape[2], input_bands.shape[3]
            break
        
        psum = torch.zeros(num_bands)
        psum_sq = torch.zeros(num_bands)
        running_min = torch.full((num_bands,), float('inf'), dtype=torch.float32)
        running_max = torch.full((num_bands,), float('-inf'), dtype=torch.float32)

        total_pixels = 0

        for input_bands, ground_truth in train_dataloader:
            image = torch.cat((input_bands, ground_truth), dim = 1)

            batch_pixels = image.shape[0] * image.shape[2] * image.shape[3]  # batch_size * H * W
            total_pixels += batch_pixels

            psum += image.sum(axis = [0, 2, 3])
            psum_sq += (image ** 2).sum(axis = [0, 2, 3])

            batch_min = image.amin(dim=[0, 2, 3])  # Min per band
            batch_max = image.amax(dim=[0, 2, 3])  # Max per band

            running_min = torch.minimum(running_min, batch_min)
            running_max = torch.maximum(running_max, batch_max)

        mean = psum / total_pixels
        var = (psum_sq / total_pixels) - (mean) ** 2
        std = torch.sqrt(var)

        return mean, std, running_min, running_max