
import torch
import rasterio
import lightning as L
import os

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader


class PM25Dataset(VisionDataset):
    def __init__(self, path, transform=None, is_fivecrop=False):
        super().__init__(path, transform)
        self.path = path
        self.file_list = [f for f in os.listdir(path) if f.endswith(".tif")] 
        self.transform = transform
        self.is_fivecrop = is_fivecrop
    
    def __getitem__(self, index):
        image_path = os.path.join(self.path, self.file_list[index])

        with rasterio.open(image_path) as src:
            image_tensor = torch.from_numpy(src.read())

        if self.transform:
            image_tensor = self.transform(image_tensor)

        if self.is_fivecrop:
            return image_tensor[:, :-1], image_tensor[:, -1:]
        
        return image_tensor[:-1], image_tensor[-1:]
        
            
    def __len__(self):
        return len(self.file_list)
    

class PM25DataModule(L.LightningDataModule):

    def __init__(self, root, batch_size=1, num_workers=0, train_transform=None, 
                 val_transform=None, test_transform=None, collate_fn=None):

        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.train_dataset = PM25Dataset(
            path=os.path.join(self.root, 'train'), 
            transform=self.train_transform,
            is_fivecrop=True
        )
        self.val_dataset = PM25Dataset(
            path=os.path.join(self.root, 'val'), 
            transform=self.val_transform,
            is_fivecrop=True
        )
        self.test_dataset = PM25Dataset(
            path=os.path.join(self.root, 'test'), 
            transform=self.test_transform,
            is_fivecrop=True
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
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
