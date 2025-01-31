from src.PM25Transforms import ToTensor
from src.PM25Dataset import PM25Dataset
from torchgeo.samplers import PreChippedGeoSampler
from torch.utils.data import DataLoader
import torch

class PM25Stats:
    def __init__(self, train_path, batch_size, num_workers):
        self.train_path = train_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def compute_statistics(self):
        train_dataset = PM25Dataset(self.train_path, transforms=ToTensor())
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

        for input_bands, ground_truth in train_dataloader:
            image = torch.cat((input_bands, ground_truth), dim = 1)
            psum += image.sum(axis = [0, 2, 3])
            psum_sq += (image ** 2).sum(axis = [0, 2, 3])
        
        count = len(train_dataset) * image_height * image_width
        mean = psum / count
        var = (psum_sq / count) - (mean) ** 2
        std = torch.sqrt(var)

        return mean, std
