import os
import shutil
from sklearn.model_selection import train_test_split

random_state = 195

# Define paths
root = os.getcwd()
path = 'data/dataset_2'
input_folder = os.path.join(root, path)
output_folder = input_folder

# Create output directories
train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")
test_dir = os.path.join(output_folder, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get list of all image files
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".tif"))]

# Split the data (80% train, 10% validation, 10% test)
train_files, temp_files = train_test_split(image_files, test_size=0.2, random_state=195)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=195)

# Function to move files to the corresponding directory
def move_files(file_list, source_folder, destination_folder):
    for file_name in file_list:
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(destination_folder, file_name)
        shutil.move(src_path, dest_path)

# Move files to their respective directories
move_files(train_files, input_folder, train_dir)
move_files(val_files, input_folder, val_dir)
move_files(test_files, input_folder, test_dir)

# Print summary
print(f"Total images: {len(image_files)}")
print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
print(f"Test set: {len(test_files)} images")