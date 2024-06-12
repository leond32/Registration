import os
import glob
import nibabel as nib
import numpy as np
import cv2

def read_nii_file(file_path):
    '''Read a .nii.gz file and return a NumPy array of type uint8 with values scaled between 0 and 255.'''
    # Read the image file
    image = nib.load(file_path)
    # Get a plain NumPy array
    image = image.get_fdata()
    # Normalize the image data
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    # Scale to 0-255 and convert to uint8
    scaled_image = (normalized_image * 255).astype(np.uint8)
    #print(scaled_image.shape)
    return scaled_image

def save_image(image, file_path):
    cv2.imwrite(file_path, image)

def build_dataset(input_folder, output_folder, image_type):
    # Get all subfolders in the input folder
    subfolders = glob.glob(input_folder + '/*')
    for subfolder in subfolders:
        # Get subfolder name e.g. rawdata_normalized/sub-001
        subfolder_name = subfolder.split('/')[-1]
        # Get the last part of the subfolder name e.g. sub-001
        last_part = subfolder_name.split('\\')[-1]
        # Get all .nii.gz files in the subfolder
        image_files = glob.glob(subfolder + '/' + image_type + '/*.nii.gz')
        for image_file in image_files:
            image = read_nii_file(image_file)
            # Save image slices
            for i in range(image.shape[0]):
                image_slice = image[i, :, :]
                if not os.path.exists(output_folder + '/' + last_part + '/fixed'):
                    os.makedirs(output_folder + '/' + last_part + '/fixed')
                save_image(image_slice, output_folder + '/' + last_part + '/fixed/'+ last_part +'_slice_{:03d}_fixed.png'.format(i)) 

def main():
    # Build dataset for T1 images
    build_dataset('/vol/aimspace/projects/practical_SoSe24/segmentation/dataset-spider/rawdata_normalized', '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/dataset_2D_T1', 'T1w')
    # Build dataset for T2 images
    build_dataset('/vol/aimspace/projects/practical_SoSe24/segmentation/dataset-spider/rawdata_normalized', '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/dataset_2D_T2', 'T2w')
    
    
if __name__ == "__main__":
    main()