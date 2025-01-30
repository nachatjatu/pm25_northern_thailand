
from torchgeo.datasets import RasterDataset
import matplotlib.pyplot as plt

class PM25Dataset(RasterDataset):
    """Implementation of a custom dataset for PM2.5 data exported from GEE
    """
    filename_glob = '*_????-??-??.tif'
    filename_regex = r'^(?P<prefix>.+)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$'
    date_format = '%Y-%m-%d'
    is_image = True
    separate_files = False
    all_bands = ['pm25_tdy', 'u_component_of_wind_10m', 
                 'v_component_of_wind_10m', 'absorbing_aerosol_index', 
                 'MaxFRP', 'elevation', 'pm25_tmr']

    def __getitem__(self, index: int):
        image = super().__getitem__(index)
        input_bands = image[:-1, :, :]  
        ground_truth = image[-1, :, :].unsqueeze(0) 
        return input_bands, ground_truth