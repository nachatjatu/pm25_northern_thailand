import os
import numpy as np
import rasterio
from pprint import pprint

root = os.getcwd()

folder = os.path.join(root, 'data/dataset_2')

files = [f for f in os.listdir(folder) if f.endswith('tif')]
results = {}
print(len(files))

for file in files:
    path = os.path.join(folder, file)
    with rasterio.open(path) as src:
        bands_with_nan = []
        
        for band_index in range(1, src.count + 1): 
            band_data = src.read(band_index)
                    
            if np.isnan(band_data).any():
                band_name = src.descriptions[band_index - 1] if src.descriptions else f"Band {band_index}"
                
                bands_with_nan.append(band_name)
        
        if bands_with_nan:
            results[file] = bands_with_nan
    

pprint(results)

for file in results:
    path = os.path.join(folder, file)
    os.remove(path)
    print(f'{path} removed.')