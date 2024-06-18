import os
import glob
import nibabel as nib
import numpy as np
import cv2
from crop_npy import remove_black_border, median_filter
from PIL import Image

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

def check_image_information(image):
    '''Check if the image is valid for further processing.'''
    # Check if the image is mostly nonblack
    threshold = 20 #10
    nonblack_pixels = np.sum(median_filter(image, kernel_size=71) > threshold)
    total_pixels = image.shape[0] * image.shape[1]
    nonblack_ratio = nonblack_pixels / total_pixels
    if nonblack_ratio < 0.75: #0.6
        return False
    return True

def check_image_ratio(image):
    # aspect ratzio = width / height
    image_ratio = image.shape[1] / image.shape[0]
    #if image_ratio > 2.5 or image_ratio < 1:
    if image_ratio < 0.7: 
        return False    
    return True

def crop_to_aspect_ratio_and_resize(image, target_ratio, target_witdh):
    # Get the current size of the image
    width, height = image.size
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        # Crop the width
        new_width = int(height * target_ratio)
        offset = (width - new_width) // 2
        box = (offset, 0, width - offset, height)
    else:
        # Crop the height
        new_height = int(width / target_ratio)
        offset = (height - new_height) // 2
        box = (0, offset, width, height - offset)
    cropped_image = image.crop(box)
    target_height = int(target_witdh / target_ratio)
    resized_image = cropped_image.resize((target_witdh, target_height))
    resized_image = np.array(resized_image)
    return resized_image

def build_dataset(input_folder, output_folder, image_type):
    image_slice_sizes = []
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
                image_slice = remove_black_border(image_slice, threshold=40) #10
                if not os.path.exists(output_folder + '/' + last_part):
                    os.makedirs(output_folder + '/' + last_part)
                #if check_image_information(image_slice) and check_image_ratio(image_slice):
                if check_image_information(image_slice) and check_image_ratio(image_slice) and image_slice.shape[0] > 128 and image_slice.shape[1] > 128:
                    #image_slice = crop_to_aspect_ratio_and_resize(Image.fromarray(image_slice), 4/2, 256)
                    image_slice_sizes = image_slice_sizes + [image_slice.shape]
                    save_image(image_slice, output_folder + '/' + last_part + '/'+ last_part +'_slice_{:03d}.png'.format(i)) 
    return image_slice_sizes


def main():
    
    # Build dataset for T1 images
    sizes1 = build_dataset('D:\\Dokumente\\03_RCI\\practical\\Folder_structure\\rawdata_normalized', 'D:\\Dokumente\\03_RCI\\practical\\Folder_structure\\Datasets\\without_black_background\\dataset_2D_T1', 'T1w')
    # Build dataset for T2 images
    sizes2 = build_dataset('D:\\Dokumente\\03_RCI\\practical\\Folder_structure\\rawdata_normalized', 'D:\\Dokumente\\03_RCI\\practical\\Folder_structure\\Datasets\\without_black_background\\dataset_2D_T2', 'T2w')
    print(min(sizes1), max(sizes1))
    print(min(sizes2), max(sizes2))
    
if __name__ == "__main__":
    main()