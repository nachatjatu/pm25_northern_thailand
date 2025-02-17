
import random
import torch


class Standardize:
    """Standardizes selected image bands

    Attributes:
        mean (torch.tensor):   
            tensor containing band-specific means
        std (torch.tensor):
            tensor containing band-specific standard deviations
        std_bands (list):
            list containing indices of bands to be standardized

    """
    def __init__(self, mean, sd, std_bands, device='cpu'):
        self.mean = mean.to(device)
        self.sd = sd.to(device)
        self.std_bands = torch.tensor(std_bands, dtype=torch.long, device=device)
    
    def __call__(self, image):
        # clone image to avoid modifying original image
        std_image = image.clone()

        # standardize via the formula (x - mean) / sd
        std_image[self.std_bands, :, :] = (
            image[self.std_bands, :, :] - self.mean[self.std_bands, None, None]
        ) / self.sd[self.std_bands, None, None]
        return std_image
    

class Normalize:
    """Normalizes selected image bands using min-max normalization

    Attributes:
        min (torch.tensor):
            tensor containing band-specific minimum values
        max (torch.tensor):
            tensor containing band-specific maximum values
        norm_bands (list):
            list containing indices of bands to be normalized
    """
    def __init__(self, min, max, norm_bands, device='cpu'):
        self.min = min.to(device)
        self.max = max.to(device)
        self.norm_bands = torch.tensor(norm_bands, dtype=torch.long, device=device)

    def __call__(self, image):
        # clone image to avoid modifying original image
        print(image.device)
        norm_image = image.clone()

        # normalize via the formula (x - x_min) / (x_max - x_min)
        norm_image[self.norm_bands] = (
            image[self.norm_bands, :, :] - self.min[self.norm_bands, None, None]
        ) / (self.max[self.norm_bands, None, None] - self.min[self.norm_bands, None, None])
        
        return norm_image


class RandomRotate:
    """Randomly rotates an image by some multiple of 90 degrees

    Attributes:
        u_wind_index (int):
            index of u_wind band in image
        v_wind_index (int):
            index of v_wind band in image
    """

    def __init__(self, u_wind_index, v_wind_index):
        """
        Args:
            u_wind_index (int):
                index of u_wind band in image
            v_wind_index (int):
                index of v_wind band in image
        """

        self.u_wind_index = u_wind_index
        self.v_wind_index = v_wind_index

    def __rotate_wind__(self, image, angle):
        u_band = image[self.u_wind_index].clone()
        v_band = image[self.v_wind_index].clone()
        if angle == 0:
            return u_band, v_band
        elif angle == 90:
            print('rotate 90')
            return -v_band, u_band
        elif angle == 180:
            print('rotate 180')
            return -u_band, -v_band
        else:
            print('rotate 270')
            return v_band, -u_band

    def __call__(self, image):
        """ Applies rotation to both spatial bands and wind components """

        # Clone to avoid modifying the original tensor
        image_clone = image.clone()

        # Choose a random angle of rotation
        angle = random.choice([90, 270])  # Only rotating by 90 degrees CCW

        # Apply spatial rotation first
        image_clone = torch.rot90(image_clone, k=angle // 90, dims=[1, 2])

        # Rotate wind components after the spatial rotation
        new_u, new_v = self.__rotate_wind__(image_clone, angle)

        # Assign adjusted wind bands back into rotated image
        image_clone[self.u_wind_index] = new_u
        image_clone[self.v_wind_index] = new_v

        return image_clone


class RandomFlip:
    """Flips image horizontally, vertically, or both, randomly

    Attributes:
        u_wind_index (int):
            index of u_wind band in image
        v_wind_index (int):
            index of v_wind band in image
    """

    def __init__(self, u_wind_index, v_wind_index):
        """
        Args:
            u_wind_index (int):
                index of u_wind band in image
            v_wind_index (int):
                index of v_wind band in image
        """

        self.u_wind_index = u_wind_index
        self.v_wind_index = v_wind_index

    def __call__(self, image):
        # Clone the image to avoid modifying the original tensor
        image_clone = image.clone()

        # Choose flip direction randomly
        flip = random.choice(['none', 'horizontal', 'vertical', 'both'])

        if flip in ['horizontal', 'both']:
            print('flipping horizontally')
            image_clone = torch.flip(image_clone, [2])  # Flip along width
            image_clone[self.u_wind_index, :, :] *= -1  # Flip wind component

        if flip in ['vertical', 'both']:
            print('flipping vertically')
            image_clone = torch.flip(image_clone, [1])  # Flip along height
            image_clone[self.v_wind_index, :, :] *= -1  # Flip wind component

        return image_clone