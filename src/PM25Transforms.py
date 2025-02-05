import random
import torch

class Standardize:
    def __init__(self, mean, std, band_indices):
        self.mean = mean
        self.std = std
        self.band_indices = band_indices
    
    def __call__(self, image):
        standardized_image = image.clone()
        band_indices = torch.tensor(self.band_indices, dtype=torch.long)
        standardized_image[band_indices] = (
            image[band_indices] - self.mean[band_indices, None, None]
            ) / self.std[band_indices, None, None]
        return standardized_image
    

class Normalize:
    def __init__(self, min, max, band_indices):
        self.min = min
        self.max = max
        self.band_indices = band_indices

    def __call__(self, image):
        normalized_image = image.clone()
        band_indices = torch.tensor(self.band_indices, dtype=torch.long)
        normalized_image[band_indices, :, :] = (
            image[band_indices] - self.min[band_indices, None, None]
            ) / (self.max[band_indices, None, None] - self.min[band_indices, None, None])
        return normalized_image


class ToTensor:
    """Extracts image tensor from a GeoSampler sample dict 

    Attributes:
        sample (dict):  GeoSampler dict w/ 'image' key and tensor value
    """
    def __call__(self, sample):
        return sample['image']


class RandomRotate:
    """Rotates image by random multiple of 90 degrees

    Attributes:
        angles (list): List containing valid angles to choose from
    """
    def __init__(self, angles = [0, 90, 180, 270]):
        self.angles = angles

    def rotate_wind(self, u_wind_band, v_wind_band, angle):
        """Handles rotation of directional wind bands

        Args:
            u_wind_band (torch.tensor):     tensor of shape (height, width)
            v_wind_band (torch.tensor)):    tensor of shape (height, width)
            angle (int):                    angle of rotation

        Raises:
            Exception: If non-right angles are specified

        Returns:
            torch.tensor, torch.tensor: rotated u-band, v-band (respectively)
        """
        # prevents data overwriting
        u_clone = u_wind_band.clone()
        v_clone = v_wind_band.clone()

        # rotates wind bands CCW, swapping and inverting values as needed
        if angle == 0:
            return u_clone, v_clone
        elif angle == 90:
            return -v_clone, u_clone
        elif angle == 180:
            return -u_clone, -v_clone
        elif angle == 270:
            return v_clone, -u_clone
        else:
            raise Exception('Invalid angle - only 0, 90, 180, 270 allowed.')


    def __call__(self, image):
        input_bands = image[:-1, :, :] 
        ground_truth = image[-1, :, :].unsqueeze(0)

        # choose random angle of rotation
        angle = random.choice(self.angles)
        
        # rotate all bands spatially
        input_bands = torch.rot90(
            input_bands, k = angle // 90, dims = [1, 2])
        ground_truth = torch.rot90(
            ground_truth, k = angle // 90, dims = [1, 2])

        # handle wind band rotation
        new_u, new_v = self.rotate_wind(input_bands[1, :, :], 
                                        input_bands[2, :, :], 
                                        angle)
        input_bands[1, :, :] = new_u
        input_bands[2, :, :] = new_v

        return torch.cat((input_bands, ground_truth), dim=0)


class RandomFlip:
    """Flips image horizontally, vertically, or both, randomly

    Attributes:
        flip_prob (float): Probability of a flip
    """
    def __init__(self, flip_prob = 0.5):
        self.flip_prob = flip_prob
    
    def __call__(self, image):
        input_bands = image[:-1, :, :]  
        ground_truth = image[-1, :, :].unsqueeze(0)

        # perform horizontal flip
        if random.random() < self.flip_prob:
            input_bands = torch.flip(input_bands, [2])
            ground_truth = torch.flip(ground_truth, [2])
            input_bands[1, :, :] = -input_bands[1, :, :]

        # perform vertical flip
        if random.random() < self.flip_prob:
            input_bands = torch.flip(input_bands, [1])
            ground_truth = torch.flip(ground_truth, [1])
            input_bands[2, :, :] = -input_bands[2, :, :]
        
        return torch.cat((input_bands, ground_truth), dim=0)