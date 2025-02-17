
import os
import rasterio
import numpy as np
from pprint import pprint
import shutil
from sklearn.model_selection import train_test_split
from time import sleep

def remove_first_band(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(('.tif'))]
    for file in files:
        path = os.path.join(data_dir, file)
        temp_file = path + ".tmp.tif"
        with rasterio.open(path) as src:
            profile = src.profile
            profile.update(count=src.count - 1)  # Reduce band count

            # Read all bands except the first
            new_data = src.read(range(2, src.count + 1))

            # Write to temporary file
            with rasterio.open(temp_file, "w", **profile) as dst:
                dst.write(new_data)

        os.replace(temp_file, file)
        print(f"Updated {file}, removed first band.")
    print(f"Complete")

def move_files(file_list, source_folder, destination_folder):
    for file_name in file_list:
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(destination_folder, file_name)
        shutil.move(src_path, dest_path)


def split_data(data_dir, train_split, val_split, test_split, seed):
    if train_split + val_split + test_split != 1:
        raise Exception('Splits do not add up to 1!')
    
    # Create output directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of all image files
    files = [f for f in os.listdir(data_dir) if f.endswith(('.tif'))]

    train_files, temp_files = train_test_split(
        files, 
        test_size=test_split + val_split, 
        random_state=seed
    )

    val_files, test_files = train_test_split(
        temp_files, 
        test_size=test_split / (test_split + val_split), 
        random_state=seed
    )

    # Move files to their respective directories
    move_files(train_files, data_dir, train_dir)
    move_files(val_files, data_dir, val_dir)
    move_files(test_files, data_dir, test_dir)

    # Print summary
    print(f"Total images: {len(files)}")
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")


def recombine_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_files = [f for f in os.listdir(train_dir) if f.endswith(('.tif'))]
    val_files = [f for f in os.listdir(val_dir) if f.endswith(('.tif'))]
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.tif'))]

    # Print summary
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")

    move_files(train_files, train_dir, data_dir)
    move_files(val_files, val_dir, data_dir)
    move_files(test_files, test_dir, data_dir)

    os.rmdir(train_dir)
    os.rmdir(val_dir)
    os.rmdir(test_dir)

    print('Recombined sets')


def find_nan(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('tif')]
    results = {}

    print(f'{len(files)} files at path {data_dir}')

    for file in files:
        path = os.path.join(data_dir, file)
        with rasterio.open(path) as src:
            bands_with_nan = []
            
            for band_index in range(1, src.count + 1): 
                band_data = src.read(band_index)
                        
                if np.isnan(band_data).any():
                    band_name = (src.descriptions[band_index - 1] 
                                 if src.descriptions 
                                 else f"Band {band_index}"
                    )
                    
                    bands_with_nan.append(band_name)
            
            if bands_with_nan:
                results[file] = bands_with_nan
        
    if len(results) > 0:
        print('NaN values found')
        pprint(results)
        delete = input('Delete these files? (Y/N): ')

        if delete == 'Y':
            print('Deleting files...')
            for file in results:
                path = os.path.join(data_dir, file)
                os.remove(path)
                print(f'{path} removed.')
        else:
            print('Operation cancelled')
    else:
        print('No NaN values found')

# find_nan('./data/dataset_3/')
split_data('./data/dataset_3/', 0.7, 0.15, 0.15, 195)
# sleep(10)
# recombine_data('./data/dataset_3/')
# remove_first_band('./data/tmp')