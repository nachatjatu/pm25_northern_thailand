
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
    



# # TESTING    
# root = os.getcwd()
# path = 'data/tmp'
# folder = os.path.join(root, path)

# compute_statistics = ComputeStatistics(folder)
# mean, std = compute_statistics.compute_mean_std()

# mean[4] = 0
# std[4] = 1

# print('mean:', mean)
# print('std:', std)

# # set up image transformations
# to_tensor = ToTensor()
# normalize = transforms.Normalize(mean, std)
# flip = ImageFlip(flip_prob=0)
# rotate = ImageRotate()
# transform = transforms.Compose([to_tensor, normalize, RandomCompose([flip, rotate])])

# # set up datasets, samplers, and dataloaders

# transformed_dataset = PM25Dataset(folder, transforms = transform)
# transformed_sampler = PreChippedGeoSampler(transformed_dataset, shuffle=True)
# transformed_dataloader = DataLoader(transformed_dataset, sampler=transformed_sampler)

# original_dataset = PM25Dataset(folder, transforms = transforms.Compose([to_tensor, normalize]))
# original_sampler = PreChippedGeoSampler(original_dataset, shuffle=True)
# original_dataloader = DataLoader(original_dataset, sampler=original_sampler)

# for transformed_input, transformed_truth in transformed_dataloader:
#     for original_input, original_truth in original_dataloader:
#         print(original_input.shape, original_truth.shape)
#         # plot regular bands
#         fig, axes = plt.subplots(2, 5, figsize=(10, 8))  # 2 rows, 2 columns
#         bands = ['pm25_tdy', 'AAI', 'FRP', 'elevation', 'pm25_tmr']
#         idx = [0, 3, 4, 5, 6]
#         original = torch.cat((original_input[0], original_truth[0]), dim=0)
#         transformed = torch.cat((transformed_input[0], transformed_truth[0]), dim=0)
#         for i in range(len(idx)):
#             if i == 0:
#                 axes[0, i].set_ylabel('original')
#                 axes[1, i].set_ylabel('transformed')
#             band_name = bands[i]
#             im1 = axes[0, i].imshow(original[idx[i]], cmap='viridis')
#             axes[0, i].set_title(band_name)

#             im2 = axes[1, i].imshow(transformed[idx[i]], cmap='viridis')
#             axes[1, i].set_title(band_name)

#         plt.tight_layout()
#         plt.show()

#         # plot directional wind bands
#         fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns

#         im1 = axes[0, 0].imshow(original[1], cmap='viridis')
#         fig.colorbar(im1, ax=axes[0, 0])
#         axes[0, 0].set_title('u-band, original')

#         im2 = axes[0, 1].imshow(original[2], cmap='viridis')
#         fig.colorbar(im2, ax=axes[0, 1])
#         axes[0, 1].set_title('v-band, original')

#         im3 = axes[1, 0].imshow(transformed[1], cmap='viridis')
#         fig.colorbar(im3, ax=axes[1, 0])
#         axes[1, 0].set_title('u-band, transformed')

#         im4 = axes[1, 1].imshow(transformed[2], cmap='viridis')
#         fig.colorbar(im4, ax=axes[1, 1])
#         axes[1, 1].set_title('v-band, transformed')

#         plt.show()
#         break
#     break