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
    experiment_runs_dir = os.path.join(base_dir, 'experiment_runs_training')
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
    total_voxels = 0

    # Calculate mean
    for i, (images, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        images = images.to(device).float()
        batch_samples = images.size(0)  # batch size
        total_images += batch_samples
        total_voxels += batch_samples * images.shape[2] * images.shape[3] * images.shape[4]
        mean += images.sum(dim=[0, 2, 3, 4])

    mean /= total_voxels

    # Calculate standard deviation
    sum_of_squared_diff = 0.0
    for i, (images, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        images = images.to(device).float()
        batch_samples = images.size(0)
        sum_of_squared_diff += ((images - mean.view(1, -1, 1, 1, 1).to(device)) ** 2).sum(dim=[0, 2, 3, 4])

    std = torch.sqrt(sum_of_squared_diff / total_voxels)

    return mean.tolist(), std.tolist()

########################################################################################################################

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

########################################################################################################################

def find_file_correspondence(image_paths, id_name, slice_name):
    corresponding_file = None
    for path in image_paths:
        if id_name in path and  slice_name in path:
            corresponding_file = path
            break
    return corresponding_file

########################################################################################################################

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
        

########################################################################################################################

def resize_3d(image, target_shape):
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    resized_image = t.interpolate(image, size=target_shape, mode='trilinear', align_corners=False)
    return resized_image.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

########################################################################################################################

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

########################################################################################################################

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

########################################################################################################################
class CustomDataset_T1w_T2w(Dataset):
    def __init__(self, image_paths_T1, image_paths_T2, hparams, dataset_augmentation=False, transform=None, device="cpu"):
        """
        Args:
            image_paths_T1 (list): List of all image Paths of T1w.
            image_paths_T2 (list): List of all image Paths of T2w.
            shape: The shape of one image in the dataset.
            mean (float): The mean value for normalization.
            std (float): The standard deviation for normalization.
            transform (bool): Whether to apply the transformation.
        """
        self.image_paths_T1 = image_paths_T1
        self.image_paths_T2 = image_paths_T2
        self.transform = transform
        self.device = device
        self.image_dimension = hparams['image_dimension']
        self.random_df_creation_setting = hparams['random_df_creation_setting']
        self.modality_mixing = hparams['modality_mixing']
        self.T_weighting = hparams['T_weighting']
        self.dataset_augmentation = dataset_augmentation
        self.img_scaling_factor = hparams['img_scaling_factor']
    
    def __len__(self):
        return len(self.image_paths_T1)
    
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
                fixed_image_path = self.image_paths_T1[idx]
                corresponding_images = self.image_paths_T2
            else:
                fixed_image_path = self.image_paths_T2[idx]
                corresponding_images = self.image_paths_T1

            fixed_image_name = os.path.basename(fixed_image_path)
            moving_image_path = fixed_image_path
            for path in corresponding_images:
                if fixed_image_name == os.path.basename(path):
                    moving_image_path = path
                    break
        else:
            if self.T_weighting == 1:
                fixed_image_path = self.image_paths_T1[idx]
            elif self.T_weighting == 2:
                fixed_image_path = self.image_paths_T2[idx]
            else:
                raise ValueError('T_weighting must be 1 or 2! If modality_mixing is True, T_weighting is not used!')
            moving_image_path = fixed_image_path

        fixed_img = torch.tensor(np.load(fixed_image_path)).float()
        moving_img = torch.tensor(np.load(moving_image_path)).float()
        fixed_img = resize_3d(fixed_img, tuple(int(dim * self.img_scaling_factor) for dim in fixed_img.shape))

        moving_img = resize_3d(moving_img, fixed_img.shape)
        fixed_img, moving_img = random_crop_3d(fixed_img, moving_img, self.image_dimension)
        #moving_img = random_crop_3d(moving_img, self.image_dimension, print_once=True)
        #fixed_img = random_crop_3d(fixed_img, self.image_dimension)
        #fixed_img = resize_3d(fixed_img, self.image_dimension)
        #moving_img = resize_3d(moving_img, self.image_dimension)
        #fixed_img = center_crop_3d(fixed_img, self.image_dimension)
        #moving_img = center_crop_3d(moving_img, self.image_dimension)
        
        shape = fixed_img.squeeze(0).T.shape

        original_image = fixed_img.unsqueeze(0).to(self.device)
        to_deform_image = moving_img.unsqueeze(0).to(self.device)

        if self.dataset_augmentation:
            deformation_layer_01 = self.build_deformation_layer(shape, self.device, fixed_img_DF=True).to(self.device)
            original_image = deformation_layer_01.deform(original_image)
            to_deform_image = deformation_layer_01.deform(to_deform_image)

        deformation_layer = self.build_deformation_layer(shape, self.device, fixed_img_DF=False).to(self.device)
        deformed_image = deformation_layer.deform(to_deform_image)
        deformation_field = deformation_layer.get_deformation_field().squeeze(0).to(self.device)

        if self.transform:
            original_image = self.transform(original_image)
            deformed_image = self.transform(deformed_image)

        stacked_image = torch.cat([original_image, deformed_image], dim=0)

        '''if idx == 0:
            print('shape of stacked image: ', stacked_image.shape)
            print('shape of deformation field: ', deformation_field.shape)'''
            
        return stacked_image, deformation_field
    
########################################################################################################################

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

########################################################################################################################        

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, scheduler, device, log_dir='afhq_logs', patience=5, alpha=0.05):
    logging.info(f'Started singele precision training the model for {n_epochs} epochs...')
    # Create directories if they don't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Initialize lists to hold the loss values
    train_losses = []
    val_losses = []

    # Open CSV file for logging
    csv_path = os.path.join(log_dir, 'losses.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    start_epoch = 0
    checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f'Resuming training from epoch {start_epoch}')

    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for i, (images, deformation_field) in enumerate(train_loader):
            images = images.float().to(device)
            deformation_field = deformation_field.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, deformation_field)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
                
        avg_train_loss = train_loss / len(train_loader)
        logging.info(f'Training Loss (Epoch {epoch+1}/{n_epochs}): {avg_train_loss:.8f}')
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (images, deformation_field) in enumerate(val_loader):
                images = images.float().to(device)
                deformation_field = deformation_field.to(device)
                
                outputs = model(images)
                batch_loss = criterion(outputs, deformation_field)
                val_loss += batch_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f'Validation Loss (Epoch {epoch+1}/{n_epochs}): {avg_val_loss:.8f}')
        val_losses.append(avg_val_loss)

        # Print training and validation losses to console
        print(f'Training Loss (Epoch {epoch+1}/{n_epochs}): {avg_train_loss:.8f}')
        print(f'Validation Loss (Epoch {epoch+1}/{n_epochs}): {avg_val_loss:.8f}')  
        sys.stdout.flush()   
                
        # Log the losses to a CSV file
        with open(csv_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss},{avg_val_loss}\n')
        
        # Save model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            logging.info(f'Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.8f}')
            print(f'Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.8f}')
            epochs_without_improvement = 0
        else:
            logging.info(f'No improvement in validation loss since epoch {best_epoch}')
            print(f'No improvement in validation loss since epoch {best_epoch}')
            epochs_without_improvement += 1
            
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }, checkpoint_path)
        
        logging.info(f'Checkpoint saved at epoch {epoch+1}')
        
        # Updatde learning rate if scheduler is enabled
        if scheduler is not None:
            # Log learning rate before the scheduler step
            prev_lr = get_lr(optimizer)
            scheduler.step(avg_val_loss)
            # Log learning rate after the scheduler step
            new_lr = get_lr(optimizer)
            if new_lr != prev_lr:
                logging.info(f'Learning rate changed from {prev_lr} to {new_lr} at epoch {epoch+1}')
                print(f'Learning rate changed from {prev_lr} to {new_lr} at epoch {epoch+1}')
        
            
        # Check for early stopping
        if epochs_without_improvement >= patience:
            print('Early stopping...')
            logging.info('Early stopping...')
            return
        
        torch.cuda.empty_cache()  
          
    logging.info('Training completed successfully')
    print('Training completed successfully')
    
########################################################################################################################
    
def train_model_mixed_precision(model, train_loader, val_loader, criterion, optimizer, n_epochs, scheduler, device, log_dir='afhq_logs', patience=5, alpha=0.05):
    logging.info(f'Started mixed precision training the model for {n_epochs} epochs...')
    # Create directories if they don't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Initialize lists to hold the loss values
    train_losses = []
    val_losses = []

    # Open CSV file for logging
    csv_path = os.path.join(log_dir, 'losses.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    start_epoch = 0
    checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f'Resuming training from epoch {start_epoch}')

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    for epoch in range(start_epoch, n_epochs):
        model.train()
        train_loss = 0
        
        for i, (images, deformation_field) in enumerate(train_loader):
            images = images.float().to(device)
            deformation_field = deformation_field.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, deformation_field)
                train_loss += loss.item()
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
        avg_train_loss = train_loss / len(train_loader)
        logging.info(f'Training Loss (Epoch {epoch+1}/{n_epochs}): {avg_train_loss:.8f}')
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (images, deformation_field) in enumerate(val_loader):
                images = images.float().to(device)
                deformation_field = deformation_field.to(device)
                
                with autocast():
                    outputs = model(images)
                    batch_loss = criterion(outputs, deformation_field)
                    val_loss += batch_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f'Validation Loss (Epoch {epoch+1}/{n_epochs}): {avg_val_loss:.8f}')
        val_losses.append(avg_val_loss)

        # Print training and validation losses to console
        print(f'Training Loss (Epoch {epoch+1}/{n_epochs}): {avg_train_loss:.8f}')
        print(f'Validation Loss (Epoch {epoch+1}/{n_epochs}): {avg_val_loss:.8f}')  
        sys.stdout.flush()   
                
        # Log the losses to a CSV file
        with open(csv_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss},{avg_val_loss}\n')
        
        # Save model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            logging.info(f'Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.8f}')
            print(f'Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.8f}')
            epochs_without_improvement = 0
        else:
            logging.info(f'No improvement in validation loss since epoch {best_epoch}')
            print(f'No improvement in validation loss since epoch {best_epoch}')
            epochs_without_improvement += 1
            
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }, checkpoint_path)
        
        logging.info(f'Checkpoint saved at epoch {epoch+1}')
        
        # Update learning rate if scheduler is enabled
        if scheduler is not None:
            # Log learning rate before the scheduler step
            prev_lr = get_lr(optimizer)
            scheduler.step(avg_val_loss)
            # Log learning rate after the scheduler step
            new_lr = get_lr(optimizer)
            if new_lr != prev_lr:
                logging.info(f'Learning rate changed from {prev_lr} to {new_lr} at epoch {epoch+1}')
                print(f'Learning rate changed from {prev_lr} to {new_lr} at epoch {epoch+1}')
        
        # Check for early stopping
        if epochs_without_improvement >= patience:
            print('Early stopping...')
            logging.info('Early stopping...')
            return
        
        torch.cuda.empty_cache()  

########################################################################################################################
def get_image_paths(root_dir): 
    image_paths = []
    for subject in os.listdir(root_dir):
        subject_dir = os.path.join(root_dir, subject)
        if os.path.isdir(subject_dir) and os.listdir(subject_dir) != []:
            for filename in os.listdir(subject_dir):
                if filename.endswith(".npy"):
                    image_paths.append(os.path.join(subject_dir, filename))   
    return image_paths

########################################################################################################################

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

        for i, (images, _) in enumerate(val_loader):
            
            # Move validation data to the device
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
    
########################################################################################################################

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
    
    epsilon = 1e-8
    # Calculate the percentage of outliers
    percentage_outliers = (n_outliers / (total_data_points + epsilon)) * 100
    
    # Set plot title and labels
    plt.title(title + f' (Outliers: {percentage_outliers:.2f}%)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
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

def save_loss_plot(experiment_dir):
    
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
        
    # create the loss plot from the csv file and save it
    csv_path = os.path.join(experiment_dir, 'losses.csv')
    df = pd.read_csv(csv_path)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'images', 'loss_plot.png'))
    plt.close()
    
############################################################################################################

def plot_image_results(model, best_model_path, data_loader, experiment_dir, device, num_samples=10, slice_idx=8):
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
        
    # Load the best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        images, deformation_fields = next(iter(data_loader))
        
        # Move data to the device
        images = images.float().to(device)
        deformation_fields = deformation_fields.to(device)
        outputs = model(images)
        
        fig, axes = plt.subplots(10, num_samples, figsize=(48, 38))
        
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
            
            # Predicted displacement field of that slide (x-component)
            ax = axes[5, i]
            ax.imshow(outputs[i, 0, slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Predicted Displacement Field (x)')
            ax.axis('off')
            
            # Predicted displacement field of that slide (y-component)
            ax = axes[6, i]
            ax.imshow(outputs[i, 1, slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Predicted Displacement Field (y)')
            ax.axis('off')
            
            # Predicted displacement field of that slide (z-component)
            ax = axes[7, i]
            ax.imshow(outputs[i, 2, slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Predicted Displacement Field (z)')
            ax.axis('off')
            
            inverse_transformed_image = deform_image_3d(images[i, 1], outputs[i], device)
            inverse_transformed_image_gt = deform_image_3d(images[i, 1], deformation_fields[i], device)
            
            # Redeformed moving image
            ax = axes[8, i]
            ax.imshow(inverse_transformed_image[slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Redeformed Moving Image')
            ax.axis('off')
            
            # Redeformed moving image (GT)
            ax = axes[9, i]
            ax.imshow(inverse_transformed_image_gt[slice_idx].cpu().numpy(), cmap='gray')
            ax.title.set_text('Redeformed Moving Image (GT)')
            ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(experiment_dir,"images","results.png"))   # save the figure to file
    plt.close(fig)    # close the figure window        
            
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

############################################################################################################

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths to the training and validation data
    data_path_T1 = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/MRI-numpy-removeblack-nopadding/T1w' 
    data_path_T2 = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/MRI-numpy-removeblack-nopadding/T2w'

    # Get the experiment_runs directory
    experiment_runs_dir = get_or_create_experiment_dir(script_dir)
        
    # Get the next experiment name
    experiment_name = get_next_experiment_number(experiment_runs_dir)
    
    experiment_dir = os.path.join(experiment_runs_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    best_model_path = os.path.join(experiment_dir,'best_model.pth')
    
    log_dir = os.path.join(experiment_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up logging
    logging.basicConfig(filename=os.path.join(log_dir,'log_file.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started running the training script...')
    
    # Define the hyperparameters for dataset creation and training
    hparams = {
        'n_epochs': 100, #100
        'batch_size': 8,
        'lr': 0.001, #0.001
        'weight_decay': 1e-6, #1e-5
        'patience': 20, 
        'alpha': 0,
        'random_df_creation_setting': 2,
        'T_weighting': 2,
        'image_dimension': (32,64,64),
        'augmentation_factor': 10,
        'modality_mixing': False,
        'lr_scheduler': True,
        'img_scaling_factor': 1/2,
    }
    logging.info(f'Loaded hyperparameters: {hparams}')
    
    # add a configuration file to save the hyperparameters in the experiment directory
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        for key, value in hparams.items():
            f.write(f'{key}: {value}\n')
    

    # Get the image paths of the T1w and T2w MRI-Images for dynamic dataset creation    
    image_paths_T1 = get_image_paths(data_path_T1)
    image_paths_T2 = get_image_paths(data_path_T2)
    
    # Create unnormalized dataset for mean and std calculation
    dataset_unnormalized = CustomDataset_T1w_T2w(image_paths_T1, image_paths_T2, hparams, dataset_augmentation=False, transform=None, device=device)
    
    data_loader_unnormalized = DataLoader(dataset_unnormalized, batch_size=hparams['batch_size'], shuffle=True)
    mean, std = calculate_mean_std_from_batches(data_loader_unnormalized, num_batches=len(data_loader_unnormalized), device=device)
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        f.write(f'Initial mean: {mean}\n')
        f.write(f'Initial std: {std}\n')
    logging.info(f'Initial Mean: {mean}, and std {std}')
        
    # Create the datasets: Consistig of the original dataset and the augmented datasets
    unaugmented_dataset = CustomDataset_T1w_T2w(image_paths_T1, image_paths_T2, hparams, dataset_augmentation=False, transform=transforms.Compose([transforms.Normalize(mean[0], std[0])]), device=device)
    datasets_with_augmentation = [CustomDataset_T1w_T2w(image_paths_T1, image_paths_T2, hparams, dataset_augmentation=True, transform=transforms.Compose([transforms.Normalize(mean[0], std[0])]), device=device) for _ in range(hparams['augmentation_factor']-1)]

    datasets_with_augmentation.append(unaugmented_dataset)
    dataset = ConcatDataset(datasets_with_augmentation)
    
    logging.info(f'Loaded augmented dataset with {len(dataset)} samples')
    
    # random split of Dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    logging.info(f'Loaded dataset with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples')

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=True)
    
    plot_images(train_loader, experiment_dir, num_samples=8, slice_idx = int(hparams['image_dimension'][0]/2))
    
    
    mean, std = calculate_mean_std_from_batches(train_loader, num_batches=5, device=device)
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        f.write(f'mean after normalizing: {mean}\n')
        f.write(f'std after normalizing: {std}\n')
    logging.info(f'Mean after normalizing: {mean}, and std after normalizing{std}')
    
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

    # Define the loss function and optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    
    if hparams['lr_scheduler']:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
        logging.info(f'Initialized optimizer with learning rate {hparams["lr"]} and weight decay {hparams["weight_decay"]} and plateau lr_scheduler')
    else:
        scheduler = None
        logging.info(f'Initialized optimizer with learning rate {hparams["lr"]} and weight decay {hparams["weight_decay"]} and no lr_scheduler')
        

    # Train the model
    #train_model(model, train_loader, val_loader, criterion, optimizer, hparams['n_epochs'], scheduler, device, log_dir=experiment_dir, patience = hparams['patience'], alpha=hparams['alpha'])
    train_model_mixed_precision(model, train_loader, val_loader, criterion, optimizer, hparams['n_epochs'], scheduler, device, log_dir=experiment_dir, patience = hparams['patience'], alpha=hparams['alpha'])
    logging.info('Finished training the model')
    
    # create the loss plot from the csv file and save it
    save_loss_plot(experiment_dir)
    logging.info('Saved the loss plot')
    
    # calculate metrics on the validation set and save them in a txt file
    evaluate_model(model, val_loader, best_model_path, experiment_dir, device)
    logging.info('Saved the metrics')
    
    # plot results
    plot_image_results(model, best_model_path, val_loader, experiment_dir, device, num_samples=8)
    logging.info('Saved image of some results')
    
if __name__ == "__main__":
    main()