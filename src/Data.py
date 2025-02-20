
import torch
import rasterio
import lightning as L
import os

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader


class PM25Dataset(VisionDataset):
    """custom VisionDataset class for processing PM2.5 image datasets

    Attributes:
        path (str):
            path to folder containing data
        file_list (list):
            list of file names in root folder
        transform:
            transformation to be applied to loaded images
        band_indices (list):
            list of indices of bands to load
    
    Methods:
        compute_statistics(batch_size=1, num_workers=0):
            computes statistics over an entire dataset of images
    """

    def __init__(self, path, transform=None, band_indices=None, is_fivecrop=False):
        """
        Args:
            path (str):
                path to folder containing data
            transform (optional):
                torchvision transform to be applied to images. Defaults to None.
            band_indices (list, optional): 
                list of indices of bands to  load. Defaults to None.
        """

        super().__init__(path, transform)
        self.path = path
        self.file_list = [f for f in os.listdir(path) if f.endswith(".tif")] 
        self.transform = transform
        self.band_indices = band_indices
        self.is_fivecrop = is_fivecrop
    
    def compute_dataset_statistics(self, batch_size=1, num_workers=0):
        """computes dataset statistics over a dataset of images

        Args:
            batch_size (int):  
                how many images to load at once in a batch
            num_workers (int):  
                how many parallel processes to use

        Returns:
            (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
                tensors for band-wise mean, std, min, and max, respectively
        """
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
        psum = torch.zeros(num_bands, dtype=torch.float32)
        psum_sq = torch.zeros(num_bands, dtype=torch.float32)
        running_min = torch.full((num_bands,), float('inf'), 
                                    dtype=torch.float32)
        running_max = torch.full((num_bands,), float('-inf'), 
                                    dtype=torch.float32)

        # keep track of total pixels for averaging
        total_pixels = 0

        for inputs, outputs in dataloader:

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
        """retrieves image from dataset by index.
        This function loads the image using rasterio, applies the transform,
        then filters the bands afterwards.

        Args:
            index (int): index of image to be retrieved

        Returns:
            (torch.tensor, torch.tensor): 
                tuple of torch tensors, where the first tensor contains 
                input bands and the second tensor contains a single output band
        """
        img_path = os.path.join(self.path, self.file_list[index])

        # open image from path and convert to tensor
        with rasterio.open(img_path) as src:
            image = src.read()

        image_tensor = torch.tensor(image, dtype=torch.float32)

        # apply transformations and filter bands
        if self.transform:
            image_tensor = self.transform(image_tensor)

        if self.is_fivecrop:
            if self.band_indices:
                image_tensor = image_tensor[:, self.band_indices, :, :]

            input = image_tensor[:, :-1]
            output = image_tensor[:, -1:]
            return input, output

        else:
            if self.band_indices:
                image_tensor = image_tensor[self.band_indices, :, :]

            input = image_tensor[:-1]
            output = image_tensor[-1:]
            return input, output
            

    def __len__(self):
        return len(self.file_list)


class PM25DataModule(L.LightningDataModule):
    """Handles data loading for PM2.5 datasets

    Attributes:
        root (str):  
            path to folder containing train, val, test data folders
        batch_size (int):  
            how many images to load at once in a batch
        num_workers (int):  
            how many parallel processes to use
        band_indices (list): 
            list of indices of bands to load
        train_transform:
            transform to be applied to training images
        val_transform:
            transform to be applied to validation images
        test_transform:
            transform to be applied to test images
    
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

    def __init__(self, root, batch_size=1, num_workers=0, band_indices=None,
                 train_transform=None, val_transform=None, test_transform=None,
                 collate_fn=None):
        """
        Args:
            root (str):
                path to folder containing train, val, test data folders
            transforms (dict):
                dictionary of transforms to be applied to images loaded by 
                train, val, and test DataLoaders.
            batch_size (int, optional): 
                how many images to load at once in a batch. Defaults to 1.
            num_workers (int, optional):
                how many parallel processes to use. Defaults to 0.
            band_indices (list, optional): 
                list of indices of bands to load. Defaults to None.
            train_transform (optional):
                transform to be applied to training images. Defaults to None.
            val_transform (optional):
                transform to be applied to validation images. Defaults to None.
            test_transform (optional):
                transform to be applied to test images. Defaults to None.
        """

        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.band_indices = band_indices
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.collate_fn = collate_fn

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
        if stage not in ['fit', 'validate', 'test', None]:
            raise ValueError(
                'Invalid stage: must be "fit", "validate", "test", or None'
            )

        if stage == 'fit' or stage is None:
            self.train_dataset = PM25Dataset(
                path=os.path.join(self.root, 'train'), 
                transform=self.train_transform,
                band_indices=self.band_indices
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            self.val_dataset = PM25Dataset(
                path=os.path.join(self.root, 'val'), 
                transform=self.val_transform,
                band_indices=self.band_indices,
                is_fivecrop=True
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = PM25Dataset(
                path=os.path.join(self.root, 'test'), 
                transform=self.test_transform,
                band_indices=self.band_indices,
                is_fivecrop=True
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
