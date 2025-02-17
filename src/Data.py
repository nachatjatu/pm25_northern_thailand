
import torch
import rasterio
import lightning as L
import os

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader


class PM25Dataset(VisionDataset):
    """custom VisionDataset class for processing PM2.5 image datasets

    Attributes:
        data_dir (str):
            path to folder containing data
        file_list (list):
            list of file names in root folder
        transform:
            torchvision transform to be applied to images
        select_bands (list):
            list of indices of bands to actually load
    
    Methods:
        compute_statistics(batch_size=1, num_workers=0):
            computes statistics over an entire dataset of images
    """

    def __init__(self, data_dir, transform=None, select_bands=None):
        """
        Args:
            data_dir (str):
                path to folder containing data
            transform (optional):
                torchvision transform to be applied to images. Defaults to None.
            select_bands (list, optional): 
                list of indices of bands to actually load. Defaults to None.
        """

        super().__init__(data_dir, transform)
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".tif")] 
        self.transform = transform
        self.select_bands = select_bands
    
    def compute_statistics(self, batch_size=1, num_workers=0):
        """computes statistics over an entire dataset of images

        Args:
            batch_size (int):  
                how many images to load at once in a batch
            num_workers (int):  
                how many parallel processes to use

        Returns:
            (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
                tensors for band-wise mean, std, min, and max, respectively
        """

        # we must explicitly handle device transfer
        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        # load dataloader
        dataloader = DataLoader(
            self, 
            batch_size=batch_size, 
            num_workers=num_workers
        )

        # get total band count from first image in Dataset
        inputs, outputs = next(iter(dataloader))
        num_bands = inputs.shape[1] + outputs.shape[1]
        
        # initialize tensors for running computations
        psum = torch.zeros(num_bands, dtype=torch.float32, device=device)
        psum_sq = torch.zeros(num_bands, dtype=torch.float32, device=device)
        running_min = torch.full((num_bands,), float('inf'), 
                                    dtype=torch.float32, device=device)
        running_max = torch.full((num_bands,), float('-inf'), 
                                    dtype=torch.float32, device=device)

        # keep track of total pixels for averaging
        total_pixels = 0

        for inputs, outputs in dataloader:
            # move tensors to device
            inputs, outputs = inputs.to(device), outputs.to(device)

            # combine tensors for batch computation
            image = torch.cat((inputs, outputs), dim = 1)

            # add batch pixel count to total
            batch_pixels = image.shape[0] * image.shape[2] * image.shape[3] 
            total_pixels += batch_pixels

            # update running trackers
            psum += image.sum(axis = [0, 2, 3])
            psum_sq += (image ** 2).sum(axis = [0, 2, 3])
            running_min = torch.minimum(running_min, image.amin(dim=[0, 2, 3]))
            running_max = torch.maximum(running_max, image.amax(dim=[0, 2, 3]))

        # compute mean, var, std
        mean = psum / total_pixels
        var = (psum_sq / total_pixels) - (mean) ** 2
        std = torch.sqrt(var)

        return mean, std, running_min, running_max
    
    def __getitem__(self, index):
        """retrieves image from dataset by index

        Args:
            index (int): index of image to be retrieved

        Returns:
            (torch.tensor, torch.tensor): 
                tuple of torch tensors, where the first tensor contains 
                input bands and the second tensor contains a single output band
        """

        img_path = os.path.join(self.data_dir, self.file_list[index])

        # open image from path and convert to tensor
        with rasterio.open(img_path) as src:
            image = src.read()
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # apply transformations and filter bands
        if self.transform:
            image_tensor = self.transform(image_tensor)
        if self.select_bands:
            image_tensor = image_tensor[self.select_bands]

        # separate image into input and output bands
        input = image_tensor[:-1]
        output = image_tensor[-1:]

        return input, output
    
    def __len__(self):
        return len(self.file_list)


class PM25DataModule(L.LightningDataModule):
    """Handles data loading for PM2.5 datasets

    Attributes:
        data_dir (str):  
            path to folder containing all data
        transforms (dict): 
            dictionary of transforms to be applied to images loaded by train, 
            val, and test DataLoaders
        batch_size (int):  
            how many images to load at once in a batch
        num_workers (int):  
            how many parallel processes to use
        select_bands (list): 
            list of indices of bands to actually load
    
    Methods:
        setup(stage=None):
            loads relevant data for specified training stage
        train_dataloader():
            returns DataLoader for training data
        val_dataloader():
            returns DataLoader for validation data
        test_dataloader():
            returns DataLoader for test data
    """

    def __init__(self, data_dir, transforms, batch_size=1, num_workers=0, 
                 select_bands=None):
        """
        Args:
            data_dir (str):
                path to folder containing all data
            transforms (dict):
                dictionary of transforms to be applied to images loaded by 
                train, val, and test DataLoaders.
            batch_size (int, optional): 
                how many images to load at once in a batch. Defaults to 1.
            num_workers (int, optional):
                how many parallel processes to use. Defaults to 0.
            select_bands (list, optional): 
                list of indices of bands to actually load. Defaults to None.
        """

        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.select_bands = select_bands

    def setup(self, stage=None):
        """loads relevant data for specified training stage

        loads data according to the following stages:
            'fit'       ->  load train and val datasets
            'validate'  ->  load val dataset
            'test'      ->  load test dataset
            None        ->  load train, val, and test dataset

        Args:
            stage (str, optional):
                string of training stage. Defaults to None.
        """

        if stage == 'fit' or stage is None:
            self.train_dataset = PM25Dataset(
                os.path.join(self.data_dir, 'train'), 
                self.transforms['train'],
                self.select_bands
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            self.val_dataset = PM25Dataset(
                os.path.join(self.data_dir, 'val'), 
                self.transforms['val'],
                self.select_bands
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = PM25Dataset(
                os.path.join(self.data_dir, 'test'), 
                self.transforms['test'],
                self.select_bands
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
