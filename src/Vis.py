import numpy as np
import matplotlib.pyplot as plt

def vis_wind(ax, u_band, v_band):
    nx, ny = 256, 256  # Grid size
    x, y = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    ax.set_aspect('equal')

    step = 10

    x_sample, y_sample = x[::step, ::step], y[::step, ::step]
    u_sample, v_sample = u_band[::step, ::step], v_band[::step, ::step]
    
    speed = np.sqrt(u_sample**2 + v_sample**2)

    ax.quiver(x_sample, y_sample, u_sample, v_sample, speed, scale = 10, cmap='inferno', width=0.0025)
    ax.axis('off')
    return ax

def vis_band(ax, band):
    ax_img = ax.imshow(band, cmap='inferno')
    ax.axis('off')
    return ax_img

def vis_all(image):
    num_bands =  image.shape[0]
    for band in range(num_bands):
        _, ax = plt.subplots(1, 1)
        vis_band(ax, image[band])
        ax.set_title(f'band {band}')

def vis_transform(original_image, transform, u_band_idx, v_band_idx):
    num_bands =  original_image.shape[0]
    transformed_image = transform(original_image)

    scalar_bands = [band for band in range(num_bands) if band != u_band_idx and band != v_band_idx]

    _, axes = plt.subplots(2, 2, figsize=(4, 4))
    vis_band(axes[0, 0], original_image[u_band_idx])
    axes[0, 0].set_title(f'u-wind (original)')
    vis_band(axes[0, 1], original_image[v_band_idx])
    axes[0, 1].set_title(f'v-wind (original)')
    vis_band(axes[1, 0], transformed_image[u_band_idx])
    axes[1, 0].set_title(f'u-wind (transformed)')
    vis_band(axes[1, 1], transformed_image[v_band_idx])
    axes[1, 1].set_title(f'v-wind (transformed)')
    plt.tight_layout()
    plt.show()
    
    for band in scalar_bands:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        original_band = original_image[band]
        transformed_band = transformed_image[band]
        vis_band(axes[0], original_band)
        axes[0].set_title(f'Band {band} (original)')
        vis_band(axes[1], transformed_band)
        axes[1].set_title(f'Band {band} (transformed)')
        plt.tight_layout()
        plt.show()


