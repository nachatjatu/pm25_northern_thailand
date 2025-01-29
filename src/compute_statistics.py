import rasterio
import numpy as np
import glob

class ComputeStatistics:
    def __init__(self, path):
        self.path = path
    
    def compute_mean_std(self):
        # Path to the folder containing GeoTIFF files
        # folder_path = os.path.join(root, path)
        tiff_files = glob.glob(f"{self.path}/*.tif")

        # Initialize accumulators for sum and sum of squares
        band_sums = None
        band_sums_of_squares = None
        total_pixels = 0

        for file in tiff_files:
            with rasterio.open(file) as src:
                data = src.read()  # Shape: [bands, height, width]
                
                # Exclude NoData values if present
                nodata_value = src.nodatavals[0]
                data = np.where(data == nodata_value, np.nan, data)

                # Initialize accumulators for the first file
                if band_sums is None:
                    band_sums = np.nansum(data, axis=(1, 2))
                    band_sums_of_squares = np.nansum(data**2, axis=(1, 2))
                else:
                    band_sums += np.nansum(data, axis=(1, 2))
                    band_sums_of_squares += np.nansum(data**2, axis=(1, 2))

                total_pixels += np.count_nonzero(~np.isnan(data[0]))  # Pixels per band

        # Compute mean and standard deviation per band
        mean = band_sums / total_pixels
        std = np.sqrt((band_sums_of_squares / total_pixels) - (mean**2))

        return mean, std