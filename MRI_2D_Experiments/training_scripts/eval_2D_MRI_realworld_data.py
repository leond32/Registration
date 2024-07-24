import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from deepali.core import functional as U
import random
from torch import nn, optim
import os
import cv2
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as t
import sys
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import re
import logging 

########################################################################################################################

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
from src.FreeFormDeformation import DeformationLayer
from networks.diffusion_unet import Unet

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

def calculate_mean_std_from_batches(data_loader, num_batches=20, device='cpu'):
    mean = 0.0
    std = 0.0
    total_images = 0

    # Calculate mean
    for i, images in enumerate(data_loader):
        
        if i >= num_batches:
            break
        
        # Remove samples with None values
        valid_indices = [j for j in range(images.size(0)) if images[j] is not None]
        if not valid_indices:
            continue
        images = images[valid_indices]
        
        images = images.to(device).float()
        batch_samples = images.size(0)  # batch size
        total_images += batch_samples
        mean += images.mean([0, 2, 3]) * batch_samples

    mean /= total_images

    # Calculate standard deviation
    sum_of_squared_diff = 0.0
    for i, images in enumerate(data_loader):
        if i >= num_batches:
            break
        images = images.to(device).float()
        batch_samples = images.size(0)
        sum_of_squared_diff += ((images - mean.view(1, -1, 1, 1).to(device)) ** 2).sum([0, 2, 3])

    std = torch.sqrt(sum_of_squared_diff / (total_images * images.shape[2] * images.shape[3]))

    return mean.tolist(), std.tolist()

############################################################################################################

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

############################################################################################################

def get_best_model_path(script_dir):
    # Navigate to the parent directory of the script directory
    base_dir = os.path.dirname(script_dir)

    # Construct the path to the best model file
    best_model_path = os.path.join(base_dir, 'model_for_eval', 'best_model.pth')

    return best_model_path

#############################################################################################################

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

############################################################################################################

def find_file_correspondence(image_paths, id_name, slice_name):
    corresponding_file = None
    for path in image_paths:
        if id_name in path and slice_name in path:
            corresponding_file = path
            break
    return corresponding_file
    
############################################################################################################

def plot_images(data_loader, experiment_dir, num_samples=10):
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
        
    # display some Fixed, Moving and DF images
    data_iter = iter(data_loader)
    images, deformation_fields = next(data_iter)
      
    fig, axes = plt.subplots(5, num_samples, figsize=(24, 16))
    for i in range(num_samples):
        ax = axes[0, i]
        ax.imshow(images[i,0,:,:].cpu().numpy(), cmap='gray') # Original image
        ax.title.set_text('Original Image')
        ax.axis('off')

        ax = axes[1, i]
        ax.imshow(images[i,1].cpu().numpy(), cmap='gray') # Deformed image
        ax.title.set_text('Deformed Image')
        ax.axis('off')

        ax = axes[2, i] 
        ax.imshow(deformation_fields[i, 0].cpu().numpy(), cmap='gray')  # X-component of the field
        ax.title.set_text('Displacement Field Axis 0')
        ax.axis('off')

        ax = axes[3, i] 
        ax.imshow(deformation_fields[i, 1].cpu().numpy(), cmap='gray')  # Y-component of the field
        ax.title.set_text('Displacement Field Axis 1')
        ax.axis('off')
        
        # not gray cmap
        ax = axes[4, i]
        ax.imshow(abs(images[i, 0].cpu().numpy() - images[i, 1].cpu().numpy()), cmap= 'jet')
        ax.title.set_text('Difference Fixed and Moving Image')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(experiment_dir,"images","fixed_moving_DF.png"))   # save the figure to file
    plt.close(fig)    # close the figure window
    

############################################################################################################

class CustomTestDataset(Dataset):
    def __init__(self, image_paths_1, image_paths_2, hparams, transform=None, device="cpu"):
        """
        Args:
            image_paths_1 (list): List of all image Paths of first scanner.
            image_paths_2 (list): List of all image Paths of second scanner.
            hparams: Dictionary of hyperparameters
        """
        self.image_paths_1 = image_paths_1
        self.image_paths_2 = image_paths_2
        self.transform = transform
        self.device = device
        self.image_dimension = hparams['image_dimension']
    def __len__(self):
        return min(len(self.image_paths_1), len(self.image_paths_2))


    def __getitem__(self, idx):
        
        '''# Randomly chose a Siemens or Philips for the fixed image path
        if random.choice([True, False]):
            fixed_image_path = self.image_paths_1[idx] # path of 1 phillips image
            corresponding_images = self.image_paths_2 # paths of all simens images
        else:
            fixed_image_path = self.image_paths_1[idx]
            corresponding_images = self.image_paths_1

        # Find the corresponding moving image
        # 1: get img path with corresponding id
        # 2: check for same basename 
        path_name_tuple = splitall(fixed_image_path)
        id_name = path_name_tuple[9] #/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/processed_png_real_world_2D/T2/ID10
        slice_name = path_name_tuple[-1]
        moving_image_path = find_file_correspondence(corresponding_images, id_name, slice_name)
        if moving_image_path is None:
            return None'''
        # Randomly choose a Siemens or Philips for the fixed image path
        if random.choice([True, False]):
            fixed_image_path = self.image_paths_1[idx] # path of 1 Philips image
            corresponding_images = self.image_paths_2 # paths of all Siemens images
        else:
            fixed_image_path = self.image_paths_2[idx]
            corresponding_images = self.image_paths_1

        # Find the corresponding moving image
        path_name_tuple = splitall(fixed_image_path) 
        id_name = path_name_tuple[-3] # Assumes ID10 is the third from the last in the path
        slice_name = path_name_tuple[-1]
        moving_image_path = find_file_correspondence(corresponding_images, id_name, slice_name)
        if moving_image_path is None:
            return None
                
        # fetch the fixed image
        fixed_img = cv2.imread(fixed_image_path)
        if fixed_img is None:
            raise FileNotFoundError(f"Image not found at path: {fixed_image_path}")
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2GRAY)
        fixed_img = Image.fromarray(fixed_img)
        
        # fetch the moving image
        moving_img = cv2.imread(moving_image_path)
        if moving_img is None:
            raise FileNotFoundError(f"Image not found at path: {moving_image_path}")

        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)
        moving_img = Image.fromarray(moving_img)
            
        # define transformations to apply to the images
        buffer_img = F.pil_to_tensor(fixed_img)
        fixed_img_shape = buffer_img.squeeze(0).shape
        transform = transforms.Resize(fixed_img_shape)
        transform2 = transforms.CenterCrop(self.image_dimension)
        
        # images need have the same shape
        moving_img = transform(moving_img)
        # apply the same random crop to the fixed and moving image
        fixed_img = transform2(fixed_img)
        moving_img = transform2(moving_img)
        
        # transform to tensor
        fixed_img = F.pil_to_tensor(fixed_img).float()
        moving_img = F.pil_to_tensor(moving_img).float()
        
        # transform the images
        if self.transform:
            fixed_img = self.transform(fixed_img)
            moving_img = self.transform(moving_img)

        # Stack the original and deformed images along the channel dimension
        stacked_image = torch.cat([fixed_img, moving_img], dim=0).squeeze(0)

        return stacked_image

############################################################################################################
def early_stopping(val_losses, patience=5):
    if len(val_losses) < patience:
        return False
    for i in range(1, patience+1):
        if val_losses[-i] < val_losses[-i-1]:
            return False
    return True

############################################################################################################        

def gradient_regularization_loss(deformation_field):
    # compute the gradient in x and y direction
    dx = deformation_field[:, :, 1:] - deformation_field[:, :, :-1]
    dy = deformation_field[:, 1:, :] - deformation_field[:, :-1, :]
    
    # compute the mean of the absolute values of the gradients
    loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    return loss

############################################################################################################        

def early_stopping(val_losses, patience=5):
    if len(val_losses) < patience:
        return False
    for i in range(1, patience+1):
        if val_losses[-i] < val_losses[-i-1]:
            return False
    return True

############################################################################################################        

def gradient_regularization_loss(deformation_field):
    # compute the gradient in x and y direction
    dx = deformation_field[:, :, 1:] - deformation_field[:, :, :-1]
    dy = deformation_field[:, 1:, :] - deformation_field[:, :-1, :]
    
    # compute the mean of the absolute values of the gradients
    loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    return loss


############################################################################################################    

def get_image_paths(root_dir): 
    
    image_paths = []
    for subject in os.listdir(root_dir):
        subject_dir = os.path.join(root_dir, subject)
        if os.path.isdir(subject_dir) and os.listdir(subject_dir) != []:
            for filename in os.listdir(subject_dir):
                if filename.endswith(".png"):
                    image_paths.append(os.path.join(subject_dir, filename))   
    return image_paths


############################################################################################################

def deform_image(deformed_image: torch.Tensor, displacement_field: torch.Tensor, device) -> torch.Tensor:
    """
    Deform a grayscale image using the given displacement field.

    Args:
        deformed_image (torch.Tensor): Grayscale image of shape (H, W).
        displacement_field (torch.Tensor): Displacement field of shape (2, H, W).

    Returns:
        torch.Tensor: Deformed image of shape (H, W).
    """
    # Ensure the input image and displacement field are on the same device
    deformed_image = deformed_image.to(device)
    displacement_field = displacement_field.to(device)
    
    # invert the displacement field
    displacement_field = -displacement_field

    # Create grid coordinates
    H, W = deformed_image.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack([grid_x, grid_y], dim=0).float().to(device)  # Shape: (2, H, W)

    # Add displacement field to grid
    new_grid = grid
    new_grid = new_grid.permute(1, 2, 0).unsqueeze(0)  # Shape: (1, H, W, 2)

    # Normalize grid values to be in the range [-1, 1]
    new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
    new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
    new_grid = new_grid + displacement_field.permute(1, 2, 0).unsqueeze(0)

    # Interpolate original image using the new grid
    deformed_image = deformed_image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    deformed_image = t.grid_sample(deformed_image, new_grid, mode='bilinear', padding_mode='border')
    return deformed_image.squeeze(0).squeeze(0)

############################################################################################################

def plot_results(model, best_model_path, data_loader, experiment_dir, device, num_samples=10):
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
        
    model.load_state_dict(torch.load(best_model_path, map_location=device))    
    model.eval()
    with torch.no_grad():
        images = next(iter(data_loader))
        # Move data to the device
        images = images.float().to(device)
    
        outputs = model(images)
    
        # Plot the original and deformed images
        fig, axes = plt.subplots(7, num_samples, figsize=(48, 20))
        for i in range(num_samples):

            ax = axes[0, i] # [0, i]
            ax.imshow(images[i, 0].cpu().numpy(), cmap='gray')
            ax.title.set_text('Original Image')
            ax.axis('off')

            ax = axes[1, i] # [1, i]
            ax.imshow(images[i, 1].cpu().numpy(), cmap='gray')
            ax.title.set_text('Deformed Image')
            ax.axis('off')

            ax = axes[2, i] # [2, i]
            ax.imshow(outputs[i, 0].cpu().numpy(), cmap='gray')
            ax.title.set_text('Pred. Displacement X')
            ax.axis('off')
        
            ax = axes[3, i]      
            ax.imshow(outputs[i, 1].cpu().numpy(), cmap='gray')
            ax.title.set_text('Pred. Displacement Y')
            ax.axis('off')
        
            inverse_transformed_image = deform_image(images[i, 1], outputs[i], device)
        
            ax = axes[4, i]
            ax.imshow(inverse_transformed_image.cpu().numpy(), cmap='gray')
            ax.title.set_text('Redef. Img (rM)')
            ax.axis('off')
        
            # not gray cmap
            ax = axes[5, i]
            ax.imshow(abs(images[i, 0].cpu().numpy() - images[i, 1].cpu().numpy()), cmap= 'jet')
            ax.title.set_text('abs(F-M)')
            ax.axis('off')
        
            ax = axes[6, i]
            ax.imshow(abs(images[i, 0].cpu().numpy() - inverse_transformed_image.cpu().numpy()), cmap='jet')
            ax.title.set_text('abs(F-rM)')
            ax.axis('off')
            
        plt.tight_layout()
        fig.savefig(os.path.join(experiment_dir,"images","results.png"))   # save the figure to file
        plt.close(fig)    # close the figure window

############################################################################################################

def normalize_image(image):
    min = image.min()
    max = image.max()
    if max - min == 0:
        return image
    return (image - image.min()) / (image.max() - image.min())

############################################################################################################

def ncc(image1, image2):
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    numerator = np.sum((image1 - mean1) * (image2 - mean2))
    denominator = np.sqrt(np.sum((image1 - mean1) ** 2) * np.sum((image2 - mean2) ** 2))
    return numerator / denominator

############################################################################################################

def build_box_plot(data, title, x_label, y_label, save_path):
    
    # Create the box plot
    box_dict = plt.boxplot(data)
    
    # Collect all outliers
    all_outliers = []
    for flier in box_dict['fliers']:
        all_outliers.extend(flier.get_ydata())
    
    # Calculate the number of outliers
    n_outliers = len(all_outliers)
    
    # Calculate the total number of data points
    if isinstance(data[0], list):
        total_data_points = sum(len(d) for d in data)
    else:
        total_data_points = len(data)
    
    # Calculate the percentage of outliers
    percentage_outliers = (n_outliers / total_data_points) * 100
    
    # Set plot title and labels
    plt.title(title + f' (Outliers: {percentage_outliers:.2f}%)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
      
############################################################################################################

############################################################################################################

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
    
############################################################################################################

def compute_metrics(model,best_model_path, val_loader, device):
    
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
            # Remove samples with None values
            valid_indices = [j for j in range(images.size(0)) if images[j] is not None]
            if not valid_indices:
                continue
            images = images[valid_indices]
            
            # Move validation data to the device
            images = images.float().to(device)
            outputs = model(images)
            batch_size = images.size(0)
            total_images += batch_size
        
            for j in range(batch_size):
                
                fixed_image = images[j,0]
                moving_image = images[j, 1]
                redeformed_moving_image = deform_image(images[j, 1], outputs[j], device)

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
    
############################################################################################################

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
    
    # Save metrics in a csv file
    csv_path = os.path.join(experiment_dir, 'metrics', 'metrics.csv')
    df = pd.DataFrame({'SSIM': metrics_values_after['SSIM'], 'PSNR': metrics_values_after['PSNR'], 'MSE': metrics_values_after['MSE'], 'L1': metrics_values_after['L1'], 'NCC': metrics_values_after['NCC'], 'SSIM_before': metrics_values_before['SSIM'], 'PSNR_before': metrics_values_before['PSNR'], 'MSE_before': metrics_values_before['MSE'], 'L1_before': metrics_values_before['L1'], 'NCC_before': metrics_values_before['NCC']})
    df.to_csv(csv_path, index=False)
    
############################################################################################################
            
def main():
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths to the training and validation data
    data_path = "/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/processed_png_real_world_2D/T2"
    
    # Get the experiment_runs directory
    experiment_runs_dir = get_or_create_experiment_dir(script_dir)
        
    # Get the next experiment name
    experiment_name = get_next_experiment_number(experiment_runs_dir)
    
    experiment_dir = os.path.join(experiment_runs_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    best_model_path = get_best_model_path(script_dir)
    

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # Get the image paths for dynamic dataset creation    
    images_paths_01, images_paths_02 = get_image_paths_testset(data_path)
    
    # Define the hyperparameters
    hparams = {
        'batch_size': 32,
        'random_df_creation_setting': 2,
        'image_dimension': (256,256)
    }
    
    # add a configuration file to save the hyperparameters in the experiment directory
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        for key, value in hparams.items():
            f.write(f'{key}: {value}\n')
            
    # Create unnormailzed dataset and dataloaders
    dataset_unnormalized = CustomTestDataset(images_paths_01, images_paths_02, hparams=hparams, transform=None, device=device)
    data_loader_unnormalized = DataLoader(dataset_unnormalized, batch_size=hparams['batch_size'], shuffle=True)
    mean, std = calculate_mean_std_from_batches(data_loader_unnormalized, num_batches=len(data_loader_unnormalized), device=device)
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        f.write(f'Initial mean: {mean}\n')
        f.write(f'Initial std: {std}\n')
    logging.info(f'Initial Mean: {mean}, and std {std}')
    
    # Create the datasets and dataloaders
    dataset = CustomTestDataset(images_paths_01, images_paths_02, hparams=hparams, transform=transforms.Compose([transforms.Normalize(mean=mean[0], std=std[0])]), device=device)
    #dataset = CustomDataset(images_paths, hparams=hparams, transform=None, device=device)
    val_loader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True)
    mean, std = calculate_mean_std_from_batches(val_loader, num_batches=5, device=device)
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        f.write(f'mean after normalizing: {mean}\n')
        f.write(f'std after normalizing: {std}\n')
    logging.info(f'Mean: {mean}, and std {std} after normalizing')
    

    # Define the model
    model = Unet(
        dim=8,
        init_dim=None,
        out_dim=2,
        dim_mults=(1, 2, 4, 8),
        channels=2,
        resnet_block_groups=8,
        learned_variance=False,
        conditional_dimensions=0,
        patch_size=1,
        attention_layer=None
    )
    
    # Check if weights file exists
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    evaluate_model(model, val_loader,best_model_path, experiment_dir, device)
    
    # plot results
    plot_results(model, best_model_path, val_loader, experiment_dir, device, num_samples=10)
    
if __name__ == "__main__":
    main()