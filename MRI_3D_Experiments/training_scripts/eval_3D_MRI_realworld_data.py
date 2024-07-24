import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
import random
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.functional as F
import torch.nn.functional as t
import sys
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import logging
import argparse
import os
import re

# Automatically determine the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

#######################################################################################################################
def add_repo_root_to_sys_path(script_dir):
    # Move up two levels from the script directory to get to the root of the repository
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    sys.path.append(repo_root)
    return repo_root

########################################################################################################################
# Add the repository root to sys.path
repo_root = add_repo_root_to_sys_path(script_dir)

# import modules
from src.FreeFormDeformation3D import DeformationLayer
from networks.diffusion_unet3D import Unet
########################################################################################################################

def get_or_create_experiment_dir(script_dir):
    base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    experiment_runs_dir = os.path.join(base_dir, 'experiment_runs_eval')
    if not os.path.exists(experiment_runs_dir):
        os.makedirs(experiment_runs_dir, exist_ok=True)
    return experiment_runs_dir

########################################################################################################################

def get_next_experiment_number(experiment_runs_dir):
    experiment_numbers = []
    for dirname in os.listdir(experiment_runs_dir):
        match = re.match(r'Experiment_(\d+)', dirname)
        if match:
            experiment_numbers.append(int(match.group(1)))
    if experiment_numbers:
        return f'Experiment_{max(experiment_numbers) + 1:02d}'
    else:
        return 'Experiment_01'

########################################################################################################################

def get_best_model_path(script_dir):
    # Navigate to the parent directory of the script directory
    base_dir = os.path.dirname(script_dir)

    # Construct the path to the best model file
    best_model_path = os.path.join(base_dir, 'model_for_eval', 'best_model.pth')

    return best_model_path

#######################################################################################################################

def calculate_mean_std_from_batches(data_loader, num_batches=20, device='cpu'):
    mean = torch.zeros(2, device=device)
    std = torch.zeros(2, device=device)
    total_voxels = 0

    # Calculate mean
    for i, images in enumerate(data_loader):
        if i >= num_batches:
            break
        #print(type(images))
        images = images.to(device).float()
        batch_samples = images.size(0)  # batch size
        total_voxels += batch_samples * images.shape[2] * images.shape[3] * images.shape[4]
        mean += images.sum(dim=[0, 2, 3, 4])

    mean /= total_voxels

    # Calculate standard deviation
    sum_of_squared_diff = torch.zeros(2, device=device)
    for i, images in enumerate(data_loader):
        if i >= num_batches:
            break
        images = images.to(device).float()
        sum_of_squared_diff += ((images - mean.view(1, -1, 1, 1, 1)) ** 2).sum(dim=[0, 2, 3, 4])

    std = torch.sqrt(sum_of_squared_diff / total_voxels)

    return mean.tolist(), std.tolist()

#################################################################################

def plot_images(data_loader, experiment_dir, num_samples=8, slice_idx=16):
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
    
    images, deformation_fields = next(iter(data_loader))
        
    # Move data to the device
    images = images.float()
    deformation_fields = deformation_fields
    
    fig, axes = plt.subplots(5, num_samples, figsize=(48, 38))
    
    # overall title
    fig.suptitle(f'Image Registration Results of one Batch (Slice: {slice_idx})', fontsize=16)
    for i in range(num_samples):
        
        # Slice of the fixed image
        ax = axes[0, i]
        ax.imshow(images[i, 0, slice_idx].cpu().numpy(), cmap='gray')
        ax.title.set_text('Fixed Image')
        ax.axis('off')
        
        # Slice of the moving image
        ax = axes[1, i]
        ax.imshow(images[i, 1, slice_idx].cpu().numpy(), cmap='gray')
        ax.title.set_text('Moving Image')
        ax.axis('off')
        
        # GT displacement field of that slide (x-component)
        ax = axes[2, i]
        ax.imshow(deformation_fields[i, 0, slice_idx].cpu().numpy(), cmap='gray')
        ax.title.set_text('GT Displacement Field (x)')
        ax.axis('off')
        
        # GT displacement field of that slide (y-component)
        ax = axes[3, i]
        ax.imshow(deformation_fields[i, 1, slice_idx].cpu().numpy(), cmap='gray')
        ax.title.set_text('GT Displacement Field (y)')
        ax.axis('off')
        
        # GT displacement field of that slide (z-component)
        ax = axes[4, i]
        ax.imshow(deformation_fields[i, 2, slice_idx].cpu().numpy(), cmap='gray')
        ax.title.set_text('GT Displacement Field (z)')
        ax.axis('off')
        
    plt.tight_layout()
    fig.savefig(os.path.join(experiment_dir,"images","fixed_moving_DF.png"))   # save the figure to file
    plt.close(fig)    # close the figure window

##########################################################################

def resize_3d(image, target_shape):
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    resized_image = t.interpolate(image, size=target_shape, mode='trilinear', align_corners=False)
    return resized_image.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

##########################################################################

# Global flag to control printing
printed = False

def center_crop_3d(image, target_shape, print_once=False):
    global printed
    d, h, w = image.shape
    td, th, tw = target_shape
    
    # Convert the torch tensor to a numpy array for padding
    image_np = image.numpy()
    
    if d < td or h < th or w < tw:
        # Calculate padding if image is smaller than target shape
        d_pad = max((td - d) // 2, 0)
        h_pad = max((th - h) // 2, 0)
        w_pad = max((tw - w) // 2, 0)
        
        # Pad the numpy array
        image_np = np.pad(image_np, ((d_pad, d_pad), (h_pad, h_pad), (w_pad, w_pad)), mode='reflect')
        
        if print_once and not printed:
            print(f"Shape after padding: {image_np.shape}")

    # Convert back to torch tensor
    image = torch.tensor(image_np)
    
    # Calculate the cropping indices
    d1 = max((image.shape[0] - td) // 2, 0)
    h1 = max((image.shape[1] - th) // 2, 0)
    w1 = max((image.shape[2] - tw) // 2, 0)

    cropped_image = image[d1:d1+td, h1:h1+th, w1:w1+tw]
    cropped_image = resize_3d(cropped_image, target_shape)
    
    if print_once and not printed:
        print(f"Shape before cropping: {image.shape}")
        print(f"Crop indices (d1, h1, w1): ({d1}, {h1}, {w1})")
        print(f"Shape after cropping: {cropped_image.shape}")
        printed = True
    
    return cropped_image

#########################################################

def random_crop_3d(image1, image2, target_shape, print_once=False):
    device = image1.device
    image1 = image1.cpu().numpy()  # Ensure the image is on the CPU for numpy operations
    image2 = image2.cpu().numpy()

    d, h, w = image1.shape
    td, th, tw = target_shape
    
    if d < td or h < th or w < tw:
        # Calculate padding if image is smaller than target shape
        d_pad = max((td - d) // 2, 0)
        h_pad = max((th - h) // 2, 0)
        w_pad = max((tw - w) // 2, 0)
        
        # Pad the numpy array
        image1 = np.pad(image1, ((d_pad, d_pad), (h_pad, h_pad), (w_pad, w_pad)), mode='reflect')
        image2 = np.pad(image2, ((d_pad, d_pad), (h_pad, h_pad), (w_pad, w_pad)), mode='reflect')
        

    # Convert back to torch tensor on CPU
    image1 = torch.tensor(image1)
    image2 = torch.tensor(image2)

    # Calculate the random cropping indices
    d1 = np.random.randint(0, max(image1.shape[0] - td, 1))
    h1 = np.random.randint(0, max(image1.shape[1] - th, 1))
    w1 = np.random.randint(0, max(image1.shape[2] - tw, 1))

    cropped_image1 = image1[d1:d1+td, h1:h1+th, w1:w1+tw]
    cropped_image1 = resize_3d(cropped_image1, target_shape)
    
    cropped_image2 = image2[d1:d1+td, h1:h1+th, w1:w1+tw]
    cropped_image2 = resize_3d(cropped_image2, target_shape)
    
    return cropped_image1.to(device), cropped_image2.to(device)  # Move back to the original device

######################################################################################################

def get_image_paths_testset(root_dir):
    path1 = []
    path2 = []
    for id_folder in sorted(os.listdir(root_dir)):
        id_folder_path = os.path.join(root_dir, id_folder)
        if os.path.isdir(id_folder_path):
            subfolders = [f for f in sorted(os.listdir(id_folder_path)) if os.path.isdir(os.path.join(id_folder_path, f))]
            if len(subfolders) >= 2:
                subfolder1_path = os.path.join(id_folder_path, subfolders[0])
                subfolder2_path = os.path.join(id_folder_path, subfolders[1])
                for image_name in sorted(os.listdir(subfolder1_path)):
                    image_path = os.path.join(subfolder1_path, image_name)
                    if os.path.isfile(image_path):
                        path1.append(image_path)
                for image_name in sorted(os.listdir(subfolder2_path)):
                    image_path = os.path.join(subfolder2_path, image_name)
                    if os.path.isfile(image_path):
                        path2.append(image_path)
    return path1, path2

######################################################################################################

class CustomDataset(Dataset):
    def __init__(self, image_paths_1, image_paths_2, hparams, dataset_augmentation=False, transform=None, device="cpu"):
        """
        Args:
            image_paths_1 (list): List of all image Paths of first sensor.
            image_paths_2 (list): List of all image Paths of second sensor.
            shape: The shape of one image in the dataset.
            mean (float): The mean value for normalization.
            std (float): The standard deviation for normalization.
            transform (bool): Whether to apply the transformation.
        """
        self.image_paths_1 = image_paths_1
        self.image_paths_2 = image_paths_2
        self.transform = transform
        self.device = device
        self.image_dimension = hparams['image_dimension']
        #self.random_df_creation_setting = hparams['random_df_creation_setting']
        self.modality_mixing = hparams['modality_mixing']
        self.T_weighting = hparams['T_weighting']
        self.dataset_augmentation = dataset_augmentation
        self.img_scaling_factor = hparams['img_scaling_factor']
    
    def __len__(self):
        return len(self.image_paths_1)
    
    def build_deformation_layer(self, shape, device, fixed_img_DF=False):
        """
        Build and return a new deformation layer for each call to __getitem__.
        This method returns the created deformation layer.
        """
        deformation_layer = DeformationLayer(shape, fixed_img_DF, random_df_creation_setting=self.random_df_creation_setting)
        deformation_layer.new_deformation(device=device)
        return deformation_layer

    def __getitem__(self, idx):
        if self.modality_mixing:
            if random.choice([True, False]):
                fixed_image_path = self.image_paths_1[idx]
                moving_image_path = self.image_paths_2[idx]
            else:
                fixed_image_path = self.image_paths_2[idx]
                moving_image_path = self.image_paths_1[idx]
                
        else:
            fixed_image_path = self.image_paths_1[idx]
            moving_image_path = self.image_paths_2[idx]


        fixed_img = torch.tensor(np.load(fixed_image_path)).float()
        moving_img = torch.tensor(np.load(moving_image_path)).float()
        fixed_img = resize_3d(fixed_img, tuple(int(dim * self.img_scaling_factor) for dim in fixed_img.shape))

        moving_img = resize_3d(moving_img, fixed_img.shape)
        fixed_img, moving_img = random_crop_3d(fixed_img, moving_img, self.image_dimension)
        
        shape = fixed_img.squeeze(0).T.shape

        original_image = fixed_img.unsqueeze(0).to(self.device)
        deformed_image = moving_img.unsqueeze(0).to(self.device)

        if self.transform:
            original_image = self.transform(original_image)
            deformed_image = self.transform(deformed_image)

        stacked_image = torch.cat([original_image, deformed_image], dim=0)
        return stacked_image
    
########################################################################################################################

def normalize_image(image):
    min = image.min()
    max = image.max()
    if max - min == 0:
        return image
    return (image - image.min()) / (image.max() - image.min())

########################################################################################################################

def ncc(image1, image2):
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    numerator = np.sum((image1 - mean1) * (image2 - mean2))
    denominator = np.sqrt(np.sum((image1 - mean1) ** 2) * np.sum((image2 - mean2) ** 2))
    return numerator / denominator

########################################################################################################################

def compute_metrics(model, best_model_path, val_loader, device):
    
    # load the best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    # Compute similarity measures of image and redeformed image from validation data (SSIM, PSNR, MSE, L1)
    model.eval()
    with torch.no_grad():
        
        avg_ssim = 0
        avg_psnr = 0
        avg_mse = 0
        avg_l1 = 0
        avg_ncc = 0
        
        avg_ssim_before = 0
        avg_psnr_before = 0
        avg_mse_before = 0
        avg_l1_before = 0
        avg_ncc_before = 0
        
        ssim_values = []
        psnr_values = []
        mse_values = []
        l1_values = []
        ncc_values = []
        
        ssim_values_before = []
        psnr_values_before = []
        mse_values_before = []
        l1_values_before = []
        ncc_values_before = []
        

        total_images = 0

        for i, images in enumerate(val_loader):
            
            # Move validation data to the device
            #print("compute", type(images))
            images = images.float().to(device)
            outputs = model(images)
            batch_size = images.size(0)
            total_images += batch_size
        
            for j in range(batch_size):
                
                fixed_image = images[j,0]
                moving_image = images[j, 1]
                redeformed_moving_image = deform_image_3d(images[j, 1], outputs[j], device)

                # Normalize images before SSIM calculation
                fixed_norm = normalize_image(fixed_image.cpu().numpy())
                moving_norm = normalize_image(moving_image.cpu().numpy())
                redeformed_moving_norm = normalize_image(redeformed_moving_image.cpu().numpy())

                # SSIM
                try:
                    # fixed and redeformed moving
                    ssim_value = ssim(fixed_norm, redeformed_moving_norm, data_range=1.0)
                    ssim_values.append(ssim_value)
                    avg_ssim += ssim_value
                    
                    # fixed and moving
                    ssim_value_before = ssim(fixed_norm, moving_norm, data_range=1.0)
                    ssim_values_before.append(ssim_value_before)
                    avg_ssim_before += ssim_value_before
                except:
                    print('Error in SSIM calculation')
                
                # MSE
                try: 
                    # fixed and redeformed moving
                    mse_value = t.mse_loss(fixed_image.to(device), redeformed_moving_image.to(device))
                    mse_values.append(mse_value.item())
                    avg_mse += mse_value.item()
                    
                    # fixed and moving
                    mse_value_before = t.mse_loss(fixed_image.to(device), moving_image.to(device))
                    mse_values_before.append(mse_value_before.item())
                    avg_mse_before += mse_value_before.item()
                except:
                    print('Error in MSE calculation')
            
                # L1
                try:
                    # fixed and redeformed moving
                    l1_value = t.l1_loss(fixed_image.to(device), redeformed_moving_image.to(device))
                    l1_values.append(l1_value.item())
                    avg_l1 += l1_value.item()
                    
                    # fixed and moving
                    l1_value_before = t.l1_loss(fixed_image.to(device), moving_image.to(device))
                    l1_values_before.append(l1_value_before.item())
                    avg_l1_before += l1_value_before.item()
                except:
                    print('Error in L1 calculation')

                epsilon = 1e-10
                
                # PSNR
                try:
                    # fixed and redeformed moving
                    psnr_value = 10 * torch.log10(1 / mse_value+epsilon)
                    psnr_values.append(psnr_value.item())
                    avg_psnr += psnr_value.item()
                    
                    # fixed and moving
                    psnr_value_before = 10 * torch.log10(1 / mse_value_before+epsilon)
                    psnr_values_before.append(psnr_value_before.item())
                    avg_psnr_before += psnr_value_before.item()
                except:
                    print('Error in PSNR calculation')
                    
                # NCC
                try:
                    # fixed and redeformed moving
                    ncc_value = ncc(fixed_norm, redeformed_moving_norm)
                    ncc_values.append(ncc_value.item())
                    avg_ncc += ncc_value.item()
                
                    # fixed and moving
                    ncc_value_before = ncc(fixed_norm, moving_norm)
                    ncc_values_before.append(ncc_value_before.item())
                    avg_ncc_before += ncc_value_before.item()
                except:
                    print('Error in NCC calculation')
                
                    
        # Normalize by the total number of images processed
        avg_ssim /= total_images
        avg_psnr /= total_images
        avg_mse /= total_images
        avg_l1 /= total_images
        avg_ncc /= total_images
        
        avg_ssim_before /= total_images
        avg_psnr_before /= total_images
        avg_mse_before /= total_images
        avg_l1_before /= total_images
        avg_ncc_before /= total_images
        
        # build dictionaries 
        # average metrics values over the whole validation dataset calculated from fixed_image and redeformed_moving_image
        avg_metrics_after = {
            'SSIM': avg_ssim,
            'PSNR': avg_psnr,
            'MSE': avg_mse,
            'L1': avg_l1,
            'NCC': avg_ncc,
        }

        # average metrics values over the whole validation dataset calculated from fixed_image and moving_image
        avg_metrics_before = {
            'SSIM': avg_ssim_before,
            'PSNR': avg_psnr_before,
            'MSE': avg_mse_before,
            'L1': avg_l1_before,
            'NCC': avg_ncc_before,
        }
        
        # metrics values for every sample of the validation dataset calculated from fixed_image and redeformed_moving_image
        metrics_values_after = {
            'SSIM': ssim_values,
            'PSNR': psnr_values,
            'MSE': mse_values,
            'L1': l1_values,
            'NCC': ncc_values,
        }
        
        # metrics values for every sample of the validation dataset calculated from fixed_image and moving_image
        metrics_values_before = {
            'SSIM': ssim_values_before,
            'PSNR': psnr_values_before,
            'MSE': mse_values_before,
            'L1': l1_values_before,
            'NCC': ncc_values_before,
        }
        
        #print(f'Computed metrics on {total_images} images: SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.4f}, MSE={avg_mse:.4f}, L1={avg_l1:.4f}, NCC={avg_ncc:.4f}')
        print(f'Computed metrics on {total_images} images: SSIM={avg_metrics_after['SSIM']:.4f}, PSNR={avg_metrics_after['PSNR']:.4f}, MSE={avg_metrics_after['MSE']:.4f}, L1={avg_metrics_after['L1']:.4f}, NCC={avg_metrics_after['NCC']:.4f}')
        return avg_metrics_before, avg_metrics_after, metrics_values_before, metrics_values_after

##############################################################################################

def deform_image_3d(deformed_image: torch.Tensor, displacement_field: torch.Tensor, device) -> torch.Tensor:
    """
    Deform a 3D grayscale image using the given displacement field.

    Args:
        deformed_image (torch.Tensor): Grayscale image of shape (D, H, W).
        displacement_field (torch.Tensor): Displacement field of shape (3, D, H, W).

    Returns:
        torch.Tensor: Deformed image of shape (D, H, W).
    """
    # Ensure the input image and displacement field are on the same device
    deformed_image = deformed_image.to(device)
    displacement_field = displacement_field.to(device)
    
    # Invert the displacement field
    displacement_field = -displacement_field

    # Create grid coordinates
    D, H, W = deformed_image.shape
    grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W))
    grid = torch.stack([grid_x, grid_y, grid_z], dim=0).float().to(device)  # Shape: (3, D, H, W)

    # Add displacement field to grid
    new_grid = grid
    new_grid = new_grid.permute(1, 2, 3, 0).unsqueeze(0)  # Shape: (1, D, H, W, 3)

    # Normalize grid values to be in the range [-1, 1]
    new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
    new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
    new_grid[..., 2] = 2.0 * new_grid[..., 2] / (D - 1) - 1.0
    new_grid = new_grid + displacement_field.permute(1, 2, 3, 0).unsqueeze(0)

    # Interpolate original image using the new grid
    deformed_image = deformed_image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
    deformed_image = t.grid_sample(deformed_image, new_grid, mode='bilinear', padding_mode='border')
    return deformed_image.squeeze(0).squeeze(0) 

##################################################################################################################
    
def build_boxplot_before_after(data_before, data_after, title, x_label, y_label, save_path):
    import matplotlib.pyplot as plt

    # Create the box plot
    box_dict = plt.boxplot([data_before, data_after], tick_labels=['Before', 'After'])
    
    # Collect all data before outliers
    all_outliers_before = []
    for flier in box_dict['fliers'][:1]:  # Adjusted to properly collect outliers before
        all_outliers_before.extend(flier.get_ydata())
    
    n_outliers_before = len(all_outliers_before)
    
    # Calculate the total number of data points before
    if isinstance(data_before[0], list):
        total_data_points_before = sum(len(d) for d in data_before)
    else:
        total_data_points_before = len(data_before)
        
    # Calculate the percentage of outliers before
    epsilon = 1e-8
    percentage_outliers_before = (n_outliers_before / (total_data_points_before + epsilon)) * 100
    
    # Collect all data after outliers
    all_outliers_after = []
    for flier in box_dict['fliers'][1:]:  # Adjusted to properly collect outliers after
        all_outliers_after.extend(flier.get_ydata())
    
    n_outliers_after = len(all_outliers_after)
    
    # Calculate the total number of data points after
    if isinstance(data_after[0], list):
        total_data_points_after = sum(len(d) for d in data_after)
    else:
        total_data_points_after = len(data_after)
    
    # Calculate the percentage of outliers after
    percentage_outliers_after = (n_outliers_after / (total_data_points_after + epsilon)) * 100
    
    # Set plot title and labels
    plt.title(title + f' (Outliers Before: {percentage_outliers_before:.2f}%, Outliers After: {percentage_outliers_after:.2f}%)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    

########################################################################################################################

def plot_image_results(model, best_model_path, data_loader, experiment_dir, device, num_samples=10, slice_idx=8):
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
        
    # Load the best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        images = next(iter(data_loader))
        
        # Move data to the device
        images = images.float().to(device)
        outputs = model(images)
        
        fig, axes = plt.subplots(7, num_samples, figsize=(48, 27))
        
        # overall title
        fig.suptitle(f'Image Registration Results of one Batch (Slice: {slice_idx})', fontsize=16)
        for i in range(num_samples):
            # Slice of the fixed image
            ax = axes[0, i]
            ax.imshow(images[i, 0, slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Fixed Image')
            ax.axis('off')
            
            # Slice of the moving image
            ax = axes[1, i]
            ax.imshow(images[i, 1, slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Moving Image')
            ax.axis('off')
            
            # Predicted displacement field of that slide (x-component)
            ax = axes[2, i]
            ax.imshow(outputs[i, 0, slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Predicted Displacement Field (x)')
            ax.axis('off')
            
            # Predicted displacement field of that slide (y-component)
            ax = axes[3, i]
            ax.imshow(outputs[i, 1, slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Predicted Displacement Field (y)')
            ax.axis('off')
            
            # Predicted displacement field of that slide (z-component)
            ax = axes[4, i]
            ax.imshow(outputs[i, 2, slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Predicted Displacement Field (z)')
            ax.axis('off')
            
            inverse_transformed_image = deform_image_3d(images[i, 1], outputs[i], device)
            
            # Redeformed moving image
            ax = axes[5, i]
            ax.imshow(inverse_transformed_image[slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Redeformed Moving Image')
            ax.axis('off')
            
            # absolute difference between fixed and redeformed moving image
            ax = axes[6, i]
            ax.imshow(torch.abs(images[i, 0] - inverse_transformed_image).cpu().numpy()[slice_idx], cmap='jet')
            ax.title.set_text('Absolute Difference (Fixed - Redeformed Moving)')
            ax.axis('off')

    
    plt.tight_layout()
    fig.savefig(os.path.join(experiment_dir,"images","results.png"))   # save the figure to file
    plt.close(fig)    # close the figure window        

#################################################################################################################

def evaluate_model(model, val_loader, best_model_path, experiment_dir, device):
    # Compute similarity measures of image and redeformed image from validation data (SSIM, PSNR, MSE, L1)
    avg_metrics_before, avg_metrics_after, metrics_values_before, metrics_values_after = compute_metrics(model, best_model_path, val_loader, device)
    
    # Create a metrics directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'metrics')):
        os.makedirs(os.path.join(experiment_dir, 'metrics'))
    
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
    
    # Save the metrics in a text file
    with open(os.path.join(experiment_dir, 'metrics', 'metrics.txt'), 'w') as f:
        f.write(f'Average SSIM before: {avg_metrics_before['SSIM']}\n')
        f.write(f'Average PSNR before: {avg_metrics_before['PSNR']}\n')
        f.write(f'Average MSE before: {avg_metrics_before['MSE']}\n')
        f.write(f'Average L1 before: {avg_metrics_before['L1']}\n')
        f.write(f'Average NCC before: {avg_metrics_before['NCC']}\n')
        f.write('\n')
        f.write(f'Average SSIM after: {avg_metrics_after['SSIM']}\n')
        f.write(f'Average PSNR after: {avg_metrics_after['PSNR']}\n')
        f.write(f'Average MSE after: {avg_metrics_after['MSE']}\n')
        f.write(f'Average L1 after: {avg_metrics_after['L1']}\n')
        f.write(f'Average NCC after: {avg_metrics_after['NCC']}\n') 
    
    # Boxplot of SSIM, PSNR, MSE, L1
    
    # Boxplot of SSIM
    build_boxplot_before_after(metrics_values_before['SSIM'], metrics_values_after['SSIM'], 'SSIM', 'SSIM', 'Values', os.path.join(experiment_dir, 'images', 'ssim_boxplot.png'))
    
    # Boxplot of PSNR
    build_boxplot_before_after(metrics_values_before['PSNR'], metrics_values_after['PSNR'], 'PSNR', 'PSNR', 'Values', os.path.join(experiment_dir, 'images', 'psnr_boxplot.png'))
    
    # Boxplot of MSE
    build_boxplot_before_after(metrics_values_before['MSE'], metrics_values_after['MSE'], 'MSE', 'MSE', 'Values', os.path.join(experiment_dir, 'images', 'mse_boxplot.png'))
    
    # Boxplot of L1
    build_boxplot_before_after(metrics_values_before['L1'], metrics_values_after['L1'], 'L1', 'L1', 'Values', os.path.join(experiment_dir, 'images', 'l1_boxplot.png'))
    
    # Boxplot of NCC
    build_boxplot_before_after(metrics_values_before['NCC'], metrics_values_after['NCC'], 'NCC', 'NCC', 'Values', os.path.join(experiment_dir, 'images', 'ncc_boxplot.png'))
    
    # Save metrics in a csv file
    csv_path = os.path.join(experiment_dir, 'metrics', 'metrics.csv')
    df = pd.DataFrame({'SSIM': metrics_values_after['SSIM'], 'PSNR': metrics_values_after['PSNR'], 'MSE': metrics_values_after['MSE'], 'L1': metrics_values_after['L1'], 'NCC': metrics_values_after['NCC'], 'SSIM_before': metrics_values_before['SSIM'], 'PSNR_before': metrics_values_before['PSNR'], 'MSE_before': metrics_values_before['MSE'], 'L1_before': metrics_values_before['L1'], 'NCC_before': metrics_values_before['NCC']})
    df.to_csv(csv_path, index=False)

###################################################################################################################

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths to the training and validation data
    data_path = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/Real-World_3D/T2'
    
    # Get the image paths of the Philips and Siemens scanner    
    image_paths_1, image_paths_2 = get_image_paths_testset(data_path)
    
    # Get the experiment_runs directory
    experiment_runs_dir = get_or_create_experiment_dir(script_dir)
        
    # Get the next experiment name
    experiment_name = get_next_experiment_number(experiment_runs_dir)
    
    experiment_dir = os.path.join(experiment_runs_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    best_model_path = get_best_model_path(script_dir)
    
    log_dir = os.path.join(experiment_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up logging
    logging.basicConfig(filename=os.path.join(log_dir,'log_file.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started running the training script...')

    # Define the hyperparameters for dataset creation and training

    hparams = {
            'batch_size': 8,
            'T_weighting': 2,
            'image_dimension': (32,64,64),
            'modality_mixing': True,
            'img_scaling_factor': 1/2,
        }
    logging.info(f'Loaded hyperparameters: {hparams}')

    # add a configuration file to save the hyperparameters in the experiment directory
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        for key, value in hparams.items():
            f.write(f'{key}: {value}\n')

    # Create the dataset unnormlized
    dataset_unnorm = CustomDataset(image_paths_1, image_paths_2, hparams, dataset_augmentation=False, transform=None, device=device)
    val_loader_unnorm = DataLoader(dataset_unnorm, batch_size=hparams['batch_size'], shuffle=True)

    # Calculate mean and std fron unnormalized dataset
    mean1, std1 = calculate_mean_std_from_batches(val_loader_unnorm, num_batches=len(val_loader_unnorm), device=device)
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        f.write(f'Initial mean: {mean1}\n')
        f.write(f'Initial std: {std1}\n')
    logging.info(f'Initial mean: {mean1}, and std {std1}')
    mean_avg = (mean1[0] + mean1[1])/2
    std_avg = (std1[0] + std1[1])/2
    
    # Create the dataset normalized
    dataset = CustomDataset(image_paths_1, image_paths_2, hparams, dataset_augmentation=False, transform=transforms.Compose([transforms.Normalize(mean=mean_avg, std=std_avg)]), device=device)
    val_loader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True)

    new_mean, new_std = calculate_mean_std_from_batches(val_loader, num_batches=len(val_loader), device=device)
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        f.write(f'new_mean: {new_mean}\n')
        f.write(f'new_std: {new_std}\n')
    logging.info(f'New Mean: {new_mean}, and new std {new_std}')

    # Define the model
    model = Unet(
        dim=64,
        init_dim=None,
        out_dim= 3, 
        dim_mults=(1, 2, 4, 8),
        channels= 2,
        resnet_block_groups=8,
        learned_variance=False,
        conditional_dimensions=0,
    )
    logging.info(f'Loaded model with {sum(p.numel() for p in model.parameters())} parameters')
        
    # add model configuration to the config file
    with open(os.path.join(experiment_dir, 'config.txt'), 'a') as f:
        f.write(f'Model: {model}\n')
        
    # add the number of parameters to the config file
    with open(os.path.join(experiment_dir, 'config.txt'), 'a') as f:
        f.write(f'Number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    model.to(device)
    logging.info(f'Moved model to device {device}')

    # Check if weights file exists
    if os.path.isfile(best_model_path): #args.resume and
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info('Loaded existing weights from previous experiment')
    else:
        logging.info('No existing weights loaded')

    # calculate metrics on the validation set and save them in a txt file
    evaluate_model(model, val_loader, best_model_path, experiment_dir, device)
    logging.info('Saved the metrics')

    # plot results
    plot_image_results(model, best_model_path, val_loader, experiment_dir, device, num_samples=8)
    logging.info('Saved image of some results')

if __name__ == "__main__":
    main()