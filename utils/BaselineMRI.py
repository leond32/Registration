import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
from FreeFormDeformation import DeformationLayer
from deepali.core import functional as U
import random
from diffusion_unet import Unet
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import logging
import argparse

############################################################################################################

def calculate_mean_std_from_batches(data_loader, num_batches=20, device='cpu'):
    mean = 0.0
    std = 0.0
    total_images = 0

    # Calculate mean
    for i, (images, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        images = images.to(device).float()
        batch_samples = images.size(0)  # batch size
        total_images += batch_samples
        mean += images.mean([0, 2, 3]) * batch_samples

    mean /= total_images

    # Calculate standard deviation
    sum_of_squared_diff = 0.0
    for i, (images, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        images = images.to(device).float()
        batch_samples = images.size(0)
        sum_of_squared_diff += ((images - mean.view(1, -1, 1, 1).to(device)) ** 2).sum([0, 2, 3])

    std = torch.sqrt(sum_of_squared_diff / (total_images * images.shape[2] * images.shape[3]))

    return mean.tolist(), std.tolist()

############################################################################################################

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
        if id_name in path and  slice_name in path:
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
      
    fig, axes = plt.subplots(5, num_samples, figsize=(48, 16))
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

class CustomDataset(Dataset):
    def __init__(self, image_paths, hparams, transform=None, device = "cpu"):
        """
        Args:
            image_paths (list): List of all image Paths.
            shape: The shape of one image in the dataset.
            mean (float): The mean value for normalization.
            std (float): The standard deviation for normalization.
            transform (bool): Whether to apply the transformation.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.device = device
        self.image_dimension = hparams['image_dimension']
        self.random_df_creation_setting = hparams['random_df_creation_setting']
    
    def __len__(self):
        return len(self.image_paths)
    
    def build_deformation_layer(self, shape, device):
        """
        Build and return a new deformation layer for each call to __getitem__.
        This method returns the created deformation layer.
        """
        deformation_layer = DeformationLayer(shape, random_df_creation_setting=self.random_df_creation_setting)
        deformation_layer.new_deformation(device=device)
        return deformation_layer

    def __getitem__(self, idx):
        # Fetch the original image
        image_path = self.image_paths[idx]
        img = img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        
        # get a random crop from the image 
        transform = transforms.RandomCrop(self.image_dimension, padding=None, pad_if_needed=True)
        img = transform(img)
        img = F.pil_to_tensor(img).float()
        # shape = img.squeeze(0).shape
        shape = img.squeeze(0).T.shape
        #original_image = img.unsqueeze(0)  # Add batch dimension
        original_image= img.to(self.device)
        

        # Build a new deformation layer for the current image
        deformation_layer = self.build_deformation_layer(shape, self.device).to(self.device)

        # Apply deformation to get the deformed image
        deformed_image = deformation_layer.deform(original_image)
        # Fetch the current deformation field
        deformation_field = deformation_layer.get_deformation_field().squeeze(0).to(self.device)
        
        # transform the images
        if self.transform:
            original_image = self.transform(original_image)
            deformed_image = self.transform(deformed_image)

        # Stack the original and deformed images along the channel dimension
        stacked_image = torch.cat([original_image, deformed_image], dim=0).squeeze(0)

        return stacked_image, deformation_field
    
############################################################################################################

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
            # Randomly choose a T1w or T2w for the fixed image path
            if random.choice([True, False]):
                fixed_image_path = self.image_paths_T1[idx] # path of 1 phillips image
                corresponding_images = self.image_paths_T2 # paths of all simens images
            else:
                fixed_image_path = self.image_paths_T2[idx]
                corresponding_images = self.image_paths_T1

            # Find the corresponding moving image
            fixed_image_name = os.path.basename(fixed_image_path)
            moving_image_path = fixed_image_path  # Default to the same path if not found
            for path in corresponding_images:
                if fixed_image_name == os.path.basename(path):
                    moving_image_path = path
                    break
        
        else:
            # choose the same modality for fixed and moving image
            if self.T_weighting == 1:
                # choose T1w
                fixed_image_path = self.image_paths_T1[idx]
            elif self.T_weighting == 2:
                # choose T2w
                fixed_image_path = self.image_paths_T2[idx]
            else: 
                raise ValueError('T_weighting must be 1 or 2! If modality_mixing is True, T_weighting is not used!')
            moving_image_path = fixed_image_path
            
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
        
        # shape = img.squeeze(0).shape
        shape = fixed_img.squeeze(0).T.shape
        original_image = fixed_img.to(self.device)
        to_deform_image = moving_img.to(self.device)
        
        # if dataset augmentation is enabled, build a deformation layer that is applied to the fixed and moving image
        if self.dataset_augmentation:
            deformation_layer_01 = self.build_deformation_layer(shape, self.device, fixed_img_DF=True).to(self.device)
            # Apply deformation to get the deformed images (fixed and moving image, that are used as new datapairs)
            original_image = deformation_layer_01.deform(original_image)
            to_deform_image = deformation_layer_01.deform(to_deform_image)
        
        # Build a new deformation layer to deform the moving image (to_deform_image)
        deformation_layer = self.build_deformation_layer(shape, self.device, fixed_img_DF=False).to(self.device)

        # Apply deformation to get the deformed image
        deformed_image = deformation_layer.deform(to_deform_image)
        
        # Fetch the current deformation field
        deformation_field = deformation_layer.get_deformation_field().squeeze(0).to(self.device)
        
        # transform the images
        if self.transform:
            original_image = self.transform(original_image)
            deformed_image = self.transform(deformed_image)

        # Stack the original and deformed images along the channel dimension
        stacked_image = torch.cat([original_image, deformed_image], dim=0).squeeze(0)

        return stacked_image, deformation_field
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
        self.modality_mixing = hparams['modality_mixing']
    
    def __len__(self):
        # TO Do
        return 999

    def __getitem__(self, idx):
        
        # Randomly chose a Siemens or Philips for the fixed image path
        if random.choice([True, False]):
            fixed_image_path = self.image_paths_1[idx] # path of 1 phillips image
            corresponding_images = self.image_paths_2 # paths of all simens images
        else:
            fixed_image_path = self.image_paths_T[idx]
            corresponding_images = self.image_paths_T

        # Find the corresponding moving image
        # 1: get img path with corresponding id
        # 2: check for same basename 
        path_name_tuple = splitall(fixed_image_name)
        id_name = path_name_tuple[9]
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
            fixed_image = self.transform(fixed_image)
            moving_image = self.transform(moving_image)

        # Stack the original and deformed images along the channel dimension
        stacked_image = torch.cat([fixed_image, moving_image], dim=0).squeeze(0)

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

############################################################################################################        

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, scheduler, device, log_dir='afhq_logs', patience=5, alpha=0.05):
    logging.info(f'Started training the model for {n_epochs} epochs...')
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
            loss = criterion(outputs, deformation_field) + alpha*gradient_regularization_loss(outputs)
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
                batch_loss = criterion(outputs, deformation_field) + alpha*gradient_regularization_loss(outputs)
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
            
    logging.info('Training completed successfully')
    print('Training completed successfully')
        
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
        images, deformation_fields = next(iter(data_loader))
        # Move data to the device
        images = images.float().to(device)
        deformation_fields = deformation_fields.to(device)
    
        outputs = model(images)
    
        # Plot the original and deformed images
        fig, axes = plt.subplots(12, num_samples, figsize=(48, 38))
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
            print('Range of Pred X: ', outputs[i, 0].min(), outputs[i, 0].max())
            ax.axis('off')
        
            ax = axes[3, i]      
            ax.imshow(outputs[i, 1].cpu().numpy(), cmap='gray')
            ax.title.set_text('Pred. Displacement Y')
            ax.axis('off')
        
            ax = axes[4, i]
            ax.imshow(deformation_fields[i, 0].cpu().numpy(), cmap='gray')
            ax.title.set_text('GT Displacement X')
            print('Range of GT X: ', deformation_fields[i, 0].min(), deformation_fields[i, 0].max())
            ax.axis('off')
        
            ax = axes[5, i]
            ax.imshow(deformation_fields[i, 1].cpu().numpy(), cmap='gray')
            ax.title.set_text('GT Displacement Y')
            ax.axis('off')        
        
            inverse_transformed_image = deform_image(images[i, 1], outputs[i], device)
            inverse_transformed_image_gt = deform_image(images[i, 1], deformation_fields[i], device)
        
            ax = axes[6, i]
            ax.imshow(inverse_transformed_image.cpu().numpy(), cmap='gray')
            ax.title.set_text('Redef. Img (rM)')
            ax.axis('off')
            
            ax = axes[7, i]
            ax.imshow(inverse_transformed_image_gt.cpu().numpy(), cmap='gray')
            ax.title.set_text('Redef. Img using GT')
            ax.axis('off')
            
            edge_detected_moving_image = cv2.Canny(images[i, 1].cpu().numpy().astype(np.uint8), 200, 300)
            img = ((images[i, 0].cpu().numpy() - np.min(images[i, 0].cpu().numpy())) / (np.max(images[i, 0].cpu().numpy()) - np.min(images[i, 0].cpu().numpy())) * 255)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            overlayed_img = img.copy()
            overlayed_img[edge_detected_moving_image > 0] = [255, 0, 0]
            
            
            ax = axes[8, i]
            ax.imshow(overlayed_img)
            ax.title.set_text('fixed w/ overlayed edges of moving')
            ax.axis('off')
        
            # not gray cmap
            ax = axes[9, i]
            ax.imshow(abs(images[i, 0].cpu().numpy() - images[i, 1].cpu().numpy()), cmap= 'jet')
            ax.title.set_text('abs(F-M)')
            ax.axis('off')
        
            ax = axes[10, i]
            ax.imshow(abs(images[i, 0].cpu().numpy() - inverse_transformed_image.cpu().numpy()), cmap='jet')
            ax.title.set_text('abs(F-rM)')
            ax.axis('off')
        
            ax = axes [11, i]
            ax.imshow(abs(images[i, 0].cpu().numpy() - inverse_transformed_image_gt.cpu().numpy()), cmap='jet')
            ax.title.set_text('abs(F-rM) using GT')
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
def compute_metrics(model,best_model_path, val_loader, device):
    
    # load the best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    # Compute similarity measures of fixed image and redeformed moving image from validation data (SSIM, PSNR, MSE, L1)
    # also compute the similarity measures between fixed and moving image
    model.eval()
    with torch.no_grad():
        avg_ssim = 0
        avg_psnr = 0
        avg_mse = 0
        avg_l1 = 0
        
        avg_ssim = 0
        avg_psnr = 0
        avg_mse = 0
        avg_l1 = 0

        ssim_values = []
        psnr_values = []
        mse_values = []
        l1_values = []
    

        total_images = 0

        for i, (images, deformation_field) in enumerate(val_loader):
            # Move validation data to the device
            images = images.float().to(device)
            deformation_field = deformation_field.to(device)
        
            outputs = model(images)
        
            batch_size = images.size(0)
            total_images += batch_size
        
            for j in range(batch_size):
                redeformed_image = deform_image(images[j, 1], outputs[j], device)

                # Normalize images before SSIM calculation
                img_0_norm = normalize_image(images[j, 0].cpu().numpy())
                
                redeformed_img_norm = normalize_image(redeformed_image.cpu().numpy())
            
                # SSIM
                try:
                    ssim_value = ssim(img_0_norm, redeformed_img_norm, data_range=1.0)
                    ssim_values.append(ssim_value)
                    avg_ssim += ssim_value
                except:
                    print('Error in SSIM calculation')
                
                # MSE
                try: 
                    mse_value = t.mse_loss(images[j, 0].to(device), redeformed_image.to(device))
                    mse_values.append(mse_value.item())
                    avg_mse += mse_value.item()
                except:
                    print('Error in MSE calculation')
                
        
                # L1
                try:
                    l1_value = t.l1_loss(images[j, 0].to(device), redeformed_image.to(device))
                    l1_values.append(l1_value.item())
                    avg_l1 += l1_value.item()
                except:
                    print('Error in L1 calculation')
               
                # PSNR
                try:
                    psnr_value = 10 * torch.log10(1 / mse_value)
                    psnr_values.append(psnr_value.item())
                    avg_psnr += psnr_value.item()
                except:
                    print('Error in PSNR calculation')
                    
        # Normalize by the total number of images processed
        avg_ssim /= total_images
        avg_psnr /= total_images
        avg_mse /= total_images
        avg_l1 /= total_images
        print(f'Computed metrics on {total_images} images: SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.4f}, MSE={avg_mse:.4f}, L1={avg_l1:.4f}')
        return avg_ssim, avg_psnr, avg_mse, avg_l1, ssim_values, psnr_values, mse_values, l1_values
############################################################################################################

def compute_metrics_new(model,best_model_path, val_loader, device):
    
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

        for i, (images, deformation_fields) in enumerate(val_loader):
        
            # Move validation data to the device
            images = images.float().to(device)
            deformation_fields = deformation_fields.to(device)
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

def evaluate_model(model, val_loader, best_model_path, experiment_dir, device):
    # Compute similarity measures of image and redeformed image from validation data (SSIM, PSNR, MSE, L1)
    avg_ssim, avg_psnr, avg_mse, avg_l1, ssim_values, psnr_values, mse_values, l1_values = compute_metrics(model, best_model_path, val_loader, device)
    
    # Create a metrics directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'metrics')):
        os.makedirs(os.path.join(experiment_dir, 'metrics'))
    
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
    
    # Save the metrics in a text file
    with open(os.path.join(experiment_dir, 'metrics', 'metrics.txt'), 'w') as f:
        f.write(f'Average SSIM: {avg_ssim}\n')
        f.write(f'Average PSNR: {avg_psnr}\n')
        f.write(f'Average MSE: {avg_mse}\n')
        f.write(f'Average L1: {avg_l1}\n')
    
    # Boxplot of SSIM, PSNR, MSE, L1
    build_box_plot([ssim_values], 'SSIM', 'SSIM', 'Values', os.path.join(experiment_dir, 'images', 'ssim_boxplot.png'))
    
    # Boxplot of PSNR
    build_box_plot([psnr_values], 'PSNR', 'PSNR', 'Values', os.path.join(experiment_dir, 'images', 'psnr_boxplot.png'))
    
    # Boxplot of MSE
    build_box_plot([mse_values], 'MSE', 'MSE', 'Values', os.path.join(experiment_dir, 'images', 'mse_boxplot.png'))
    
    # Boxplot of L1
    build_box_plot([l1_values], 'L1', 'L1', 'Values', os.path.join(experiment_dir, 'images', 'l1_boxplot.png'))
    
    # Save metrics in a csv file
    csv_path = os.path.join(experiment_dir, 'metrics', 'metrics.csv')
    df = pd.DataFrame({'SSIM': ssim_values, 'PSNR': psnr_values, 'MSE': mse_values, 'L1': l1_values})
    df.to_csv(csv_path, index=False)

############################################################################################################

def evaluate_model_new(model, val_loader, best_model_path, experiment_dir, device):
    # Compute similarity measures of image and redeformed image from validation data (SSIM, PSNR, MSE, L1)
    avg_metrics_before, avg_metrics_after, metrics_values_before, metrics_values_after = compute_metrics_new(model, best_model_path, val_loader, device)
    
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
    build_box_plot([metrics_values_after['SSIM']], 'SSIM', 'SSIM', 'Values', os.path.join(experiment_dir, 'images', 'ssim_after_boxplot.png'))
    build_box_plot([metrics_values_before['SSIM']], 'SSIM', 'SSIM', 'Values', os.path.join(experiment_dir, 'images', 'ssim_before_boxplot.png'))
    
    # Boxplot of PSNR
    build_box_plot([metrics_values_after['PSNR']], 'PSNR', 'PSNR', 'Values', os.path.join(experiment_dir, 'images', 'psnr_after_boxplot.png'))
    build_box_plot([metrics_values_before['PSNR']], 'PSNR', 'PSNR', 'Values', os.path.join(experiment_dir, 'images', 'psnr_before_boxplot.png'))
    
    # Boxplot of MSE
    build_box_plot([metrics_values_after['MSE']], 'MSE', 'MSE', 'Values', os.path.join(experiment_dir, 'images', 'mse_after_boxplot.png'))
    build_box_plot([metrics_values_before['MSE']], 'MSE', 'MSE', 'Values', os.path.join(experiment_dir, 'images', 'mse_before_boxplot.png'))
    
    # Boxplot of L1
    build_box_plot([metrics_values_after['L1']], 'L1', 'L1', 'Values', os.path.join(experiment_dir, 'images', 'l1_after_boxplot.png'))
    build_box_plot([metrics_values_before['L1']], 'L1', 'L1', 'Values', os.path.join(experiment_dir, 'images', 'l1_before_boxplot.png'))
    
    # Boxplot of NCC
    build_box_plot([metrics_values_after['NCC']], 'NCC', 'NCC', 'Values', os.path.join(experiment_dir, 'images', 'ncc_after_boxplot.png'))
    build_box_plot([metrics_values_before['NCC']], 'NCC', 'NCC', 'Values', os.path.join(experiment_dir, 'images', 'ncc_before_boxplot.png'))
    
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
    data_path_T1 = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/MRI_Slices_PNG/MRI_slices_diff_res/dataset_2D_T1w'
    data_path_T2 = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/MRI_Slices_PNG/MRI_slices_diff_res/dataset_2D_T2w'
    
    
    # Define the paths to save the logs and the best model	
    experiments_dir = '/vol/aimspace/projects/practical_SoSe24/registration_group/MRI_Experiments/train_with_lr_sched' # Change if you dont train on AFHQ
    experiment_name = 'Experiment_03_1' # Change this to a different name for each experiment 
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    best_model_path = os.path.join(experiment_dir,'best_model.pth')
    log_dir = os.path.join(experiment_dir, 'logs')
    
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up logging
    logging.basicConfig(filename=os.path.join(log_dir,'log_file.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started running the training script...')
    
    # Define the hyperparameters for dataset creation and training

    hparams = {
        'mean': 116.37,
        'std': 78.5,
        'n_epochs': 200,
        'batch_size': 32,
        'lr': 0.001, #0.001
        'weight_decay': 1e-5,
        'patience': 30, 
        'alpha': 0,
        'random_df_creation_setting': 2,
        'T_weighting': 2,
        'image_dimension': (128,128),
        'augmentation_factor': 10,
        'modality_mixing': False,
        'lr_scheduler': True,
    }
    logging.info(f'Loaded hyperparameters: {hparams}')
    
    # add a configuration file to save the hyperparameters in the experiment directory
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        for key, value in hparams.items():
            f.write(f'{key}: {value}\n')
    

    # Get the image paths of the T1w and T2w MRI-Images for dynamic dataset creation    
    image_paths_T1 = get_image_paths(data_path_T1)
    image_paths_T2 = get_image_paths(data_path_T2)
        
    # Create the datasets: Consistig of the original dataset and the augmented datasets
    unaugmented_dataset = CustomDataset_T1w_T2w(image_paths_T1, image_paths_T2, hparams, dataset_augmentation=False, transform=transforms.Compose([transforms.Normalize(mean=[hparams['mean']], std=[hparams['std']])]), device=device)
    datasets_with_augmentation = [CustomDataset_T1w_T2w(image_paths_T1, image_paths_T2, hparams, dataset_augmentation=True, transform=transforms.Compose([transforms.Normalize(mean=[hparams['mean']], std=[hparams['std']])]), device=device) for _ in range(hparams['augmentation_factor']-1)]
    datasets_with_augmentation.append(unaugmented_dataset)
    dataset = ConcatDataset(datasets_with_augmentation)
    logging.info(f'Loaded augmented dataset with {len(dataset)} samples')
    #dataset = CustomDataset(images_paths, hparams=hparams, transform=None, device=device)
    
    # random split of Dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    logging.info(f'Loaded dataset with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples')

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=True)
    
    plot_images(train_loader, experiment_dir, num_samples=16)
    
    '''mean, std = calculate_mean_std_from_batches(train_loader, num_batches=500, device=device)
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        f.write(f'mean: {mean}\n')
        f.write(f'std: {std}\n')
    print('mean: ', mean)
    print('std: ',std)
    logging.info(f'Mean: {mean}, and std {std}')'''
    
    # Define the model
    model = Unet(
        dim=32,
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
    train_model(model, train_loader, val_loader, criterion, optimizer, hparams['n_epochs'], scheduler, device, log_dir=experiment_dir, patience = hparams['patience'], alpha=hparams['alpha'])
    logging.info('Finished training the model')
    
    # create the loss plot from the csv file and save it
    save_loss_plot(experiment_dir)
    logging.info('Saved the loss plot')
    
    # calculate metrics on the validation set and save them in a txt file
    evaluate_model_new(model, val_loader, best_model_path, experiment_dir, device)
    logging.info('Saved the metrics')
    
    # plot results
    plot_results(model, best_model_path, val_loader, experiment_dir, device, num_samples=16)
    logging.info('Saved image of some results')
    
    
    
if __name__ == "__main__":
    main()