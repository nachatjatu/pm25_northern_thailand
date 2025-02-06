from torchgeo.datasets import RasterDataset
import src.Transforms as Transforms
import torchvision.transforms as transforms
import lightning as L
from torchgeo.samplers import PreChippedGeoSampler
import os
from torch.utils.data import DataLoader
import torch

class PM25Stats:
    def __init__(self, train_path, batch_size, num_workers, transform=Transforms.ToTensor()):
        self.train_path = train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def compute_statistics(self):
        train_dataset = PM25Dataset(self.train_path, transforms=self.transform)
        train_sampler = PreChippedGeoSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=self.batch_size, num_workers=self.num_workers
        )
        
        input_bands, ground_truth = next(iter(train_dataloader))
        num_bands = input_bands.shape[1] + ground_truth.shape[1]
        
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

            batch_min = image.amin(dim=[0, 2, 3])  
            batch_max = image.amax(dim=[0, 2, 3]) 

            running_min = torch.minimum(running_min, batch_min)
            running_max = torch.maximum(running_max, batch_max)

        mean = psum / total_pixels
        var = (psum_sq / total_pixels) - (mean) ** 2
        std = torch.sqrt(var)

        return mean, std, running_min, running_max
    

class PM25Dataset(RasterDataset):
    """Implementation of a custom dataset for PM2.5 data exported from GEE
    """
    filename_glob = '*_????-??-??.tif'
    filename_regex = r'^(?P<prefix>.+)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$'
    date_format = '%Y-%m-%d'
    is_image = True
    separate_files = False

    def __init__(self, paths, crs=None, res=None, bands=None, transforms=None, cache=None, band_indices=None):
        self.band_indices = band_indices
        super().__init__(paths, crs, res, bands, transforms, cache)

    def __getitem__(self, index: int):
        image = super().__getitem__(index)
        input_bands = image[:-1, :, :]
        if self.band_indices:
            input_bands = input_bands[self.band_indices[:-1]] 
        ground_truth = image[-1, :, :].unsqueeze(0) 
        return input_bands, ground_truth
    

class PM25DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, norm_bands, std_bands, band_indices=None):
        super().__init__()
        self.data_dir = data_dir
        self.train_path = os.path.join(self.data_dir, 'train')
        self.val_path = os.path.join(self.data_dir, 'val')
        self.test_path = os.path.join(self.data_dir, 'test')

        self.band_indices = band_indices
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.norm_bands = norm_bands
        self.std_bands = std_bands

        normalize, standardize = self.prepare_transforms()
        self.transforms = transforms.Compose(
            [Transforms.ToTensor(), normalize, standardize]
        )
        
    def prepare_transforms(self):
        mean, std, min, max = PM25Stats(
            self.train_path, self.batch_size, self.num_workers).compute_statistics()
        normalize = Transforms.Normalize(min, max, self.norm_bands)
        standardize = Transforms.Standardize(mean, std, self.std_bands)
        return normalize, standardize

    def setup(self, stage=None):
        self.train_dataset = PM25Dataset(
            self.train_path, transforms=self.transforms, 
            band_indices=self.band_indices
        )
        self.val_dataset = PM25Dataset(
            self.val_path, transforms=self.transforms, 
            band_indices=self.band_indices
        )
        self.test_dataset = PM25Dataset(
            self.test_path, transforms=self.transforms, 
            band_indices=self.band_indices
        )

        self.train_sampler = PreChippedGeoSampler(self.train_dataset, shuffle=True)
        self.val_sampler = PreChippedGeoSampler(self.val_dataset, shuffle=False)
        self.test_sampler = PreChippedGeoSampler(self.test_dataset, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler 
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler 
        )