import os
import numpy as np
import rasterio
from pprint import pprint

root = os.getcwd()

folder = os.path.join(root, 'data/dataset_1')

files = [f for f in os.listdir(folder) if f.endswith('tif')]
results = {}
print(len(files))

for file in files:
    path = os.path.join(folder, file)
    with rasterio.open(path) as src:
        # Initialize an empty list to store bands with NaN
        bands_with_nan = []
        
        # Loop through each band in the GeoTIFF
        for band_index in range(1, src.count + 1): 
            band_data = src.read(band_index)
                    
            # Check if the band contains any NaN values
            if np.isnan(band_data).any():
                # Get the band name from metadata (if available)
                band_name = src.descriptions[band_index - 1] if src.descriptions else f"Band {band_index}"
                
                # Append both the index and name
                bands_with_nan.append(band_name)
        
        # Store results for this file
        if bands_with_nan:
            results[file] = bands_with_nan
    

pprint(results)

for file in results:
    path = os.path.join(folder, file)
    os.remove(path)
    print(f'{path} removed.')