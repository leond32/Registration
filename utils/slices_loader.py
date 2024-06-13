import os
import numpy as np
from TPTBox import NII
import matplotlib.pyplot as plt

import sys
# Add the Tingxuan/utils directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Tingxuan', 'utils')))
# Now you can import crop_npy
from crop_npy import remove_black_border

def process_file(file_path, save_dir, prefix):
    # Load the NIfTI file
    nifty = NII.load(file_path, seg=True)

    # Print the shape and orientation of the image
    #print(f"Processing {file_path}")
    #print("Shape:", nifty.shape)
    #print("Orientation:", nifty.orientation)

    # Get the array data from the NIfTI file
    arr = nifty.get_array()
    numpy_array = np.array(arr)

    # Define the slice range
    slice_start = 0
    slice_end = numpy_array.shape[0]

    # Slice the data
    numpy_sliced = numpy_array[slice_start:slice_end, :, :]

    # Save each slice separately
    for i in range(numpy_sliced.shape[0]):
        slice_ = numpy_sliced[i, :, :]
        slice_ = remove_black_border(slice_, threshold=5)
        slice_filename = f'{prefix}_slice_{i+1}.npy' #.npy
        slice_save_path = os.path.join(save_dir, slice_filename)
        np.save(slice_save_path, slice_)
        print(f"Saved {slice_save_path}")

def process_directory(directory_path, save_base_dir):
    # Iterate through all subdirectories and files
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)

                # Determine modality (T1 or T2)
                modality = os.path.basename(os.path.dirname(file_path))
                if modality not in ['T1w', 'T2w']:
                    continue  # Skip files that are not T1 or T2

                # Create a corresponding directory in the save_base_dir
                sub_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                save_dir = os.path.join(save_base_dir, modality, sub_dir)
                os.makedirs(save_dir, exist_ok=True)

                prefix = f"{sub_dir}_{modality}"

                process_file(file_path, save_dir, prefix)

if __name__ == "__main__":
    # Set the dataset path and the base save directory
    path_to_dataset = "/vol/aimspace/projects/practical_SoSe24/segmentation/dataset-spider/rawdata_normalized"
    save_base_dir = "/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/MRI_Slices"
    os.makedirs(save_base_dir, exist_ok=True)

    # Process all files in the dataset
    process_directory(path_to_dataset, save_base_dir)

    #print("All slices saved successfully.")