
import random
import torch


class Standardize:
    def __init__(self, means, sds, bands):
        self.means = means
        self.sds = sds
        self.bands = torch.tensor(bands, dtype=torch.int)
    
    def __call__(self, image):
        std_image = image.clone()
        std_image[self.bands] = (
            image[self.bands] - self.means[self.bands, None, None]
        ) / self.sds[self.bands, None, None]
        return std_image
    

class Normalize:
    def __init__(self, mins, maxs, bands):
        self.mins = mins
        self.maxs = maxs
        self.bands = torch.tensor(bands, dtype=torch.int)

    def __call__(self, image):
        norm_image = image.clone()
        norm_image[self.bands] = (
            image[self.bands, :, :] - self.mins[self.bands, None, None]
        ) / (self.maxs[self.bands, None, None] - self.mins[self.bands, None, None])
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