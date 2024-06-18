import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from FreeFormDeformation import DeformationLayer
from deepali.core import functional as U
import random
from diffusion_unet import Unet
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

############################################################################################################

def calculate_mean_std_from_batches(data_loader, num_batches=20):
    mean = 0.0
    std = 0.0
    total_images = 0

    # Calculate mean
    for i, (images, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        images = images.float()
        batch_samples = images.size(0)  # batch size
        total_images += batch_samples
        mean += images.mean([0, 2, 3]) * batch_samples

    mean /= total_images

    # Calculate standard deviation
    sum_of_squared_diff = 0.0
    for i, (images, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        images = images.float()
        batch_samples = images.size(0)
        sum_of_squared_diff += ((images - mean.view(1, -1, 1, 1)) ** 2).sum([0, 2, 3])

    std = torch.sqrt(sum_of_squared_diff / (total_images * images.shape[2] * images.shape[3]))

    return mean.tolist(), std.tolist()

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

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, log_dir='afhq_logs', patience=5, alpha=0.05):
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
            print(f'Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.8f}')
        

        # Check for early stopping
        if early_stopping(val_losses, patience):
            print('Early stopping...')
            break          
    
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

def plot_results(model, data_loader, experiment_dir, device, num_samples=10):
    # Create a images directory if it doesn't exist
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
        
    model.eval()
    with torch.no_grad():
        images, deformation_fields = next(iter(data_loader))
        # Move data to the device
        images = images.float().to(device)
        deformation_fields = deformation_fields.to(device)
    
        outputs = model(images)
    
        # Plot the original and deformed images
        fig, axes = plt.subplots(11, num_samples, figsize=(24, 24))
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
            ax.title.set_text('Predicted Displacement X')
            print('Range of Pred X: ', outputs[i, 0].min(), outputs[i, 0].max())
            ax.axis('off')
        
            ax = axes[3, i]      
            ax.imshow(outputs[i, 1].cpu().numpy(), cmap='gray')
            ax.title.set_text('Predicted Displacement Y')
            ax.axis('off')
        
            ax = axes[4, i]
            ax.imshow(deformation_fields[i, 0].cpu().numpy(), cmap='gray')
            ax.title.set_text('Ground Truth Displacement X')
            print('Range of GT X: ', deformation_fields[i, 0].min(), deformation_fields[i, 0].max())
            ax.axis('off')
        
            ax = axes[5, i]
            ax.imshow(deformation_fields[i, 1].cpu().numpy(), cmap='gray')
            ax.title.set_text('Ground Truth Displacement Y')
            ax.axis('off')        
        
            inverse_transformed_image = deform_image(images[i, 1], outputs[i], device)
        
            ax = axes[6, i]
            ax.imshow(inverse_transformed_image.cpu().numpy(), cmap='gray')
            ax.title.set_text('Redeformed Image')
            ax.axis('off')
        
            # not gray cmap
            ax = axes[7, i]
            ax.imshow(abs(images[i, 0].cpu().numpy() - images[i, 1].cpu().numpy()), cmap= 'jet')
            ax.title.set_text('Difference Fixed and Moving Image')
            ax.axis('off')
        
            ax = axes[8, i]
            ax.imshow(abs(images[i, 0].cpu().numpy() - inverse_transformed_image.cpu().numpy()), cmap='jet')
            ax.title.set_text('Difference Fixed and Redeformed using Pred.')
            ax.axis('off')
        
            inverse_transformed_image_gt = deform_image(images[i, 1], deformation_fields[i], device)
            ax = axes [9, i]
            ax.imshow(abs(images[i, 0].cpu().numpy() - inverse_transformed_image_gt.cpu().numpy()), cmap='jet')
            ax.title.set_text('Difference Fixed and Redeformed using GT')
            ax.axis('off')
        
            ax = axes[10, i]
            ax.imshow(inverse_transformed_image_gt.cpu().numpy(), cmap='gray')
            ax.title.set_text('Redeformed Image using GT')
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
            
def main():
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths to the training and validation data
    data_path = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/MRI_Slices_PNG/MRI_slices_diff_res/dataset_2D_T2w'
    
    
    # Define the paths to save the logs and the best model	
    log_dir = '/vol/aimspace/projects/practical_SoSe24/registration_group/MRI_Experiments/first_training' # Change if you dont train on AFHQ
    experiment_name = 'Experiment_01' # Change this to a different name for each experiment 
    experiment_dir = os.path.join(log_dir, experiment_name)
    best_model_path = os.path.join(experiment_dir,'best_model.pth')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # Get the image paths for dynamic dataset creation    
    images_paths = get_image_paths(data_path)
    
    # Define the hyperparameters
    hparams = {
        'mean': 118,
        'std': 70,
        'n_epochs': 400,
        'batch_size': 16,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'patience': 10,
        'alpha': 0,
        'random_df_creation_setting': 2,
        'T_weighting': 2,
        'image_dimension': (256,256)
    }
    
    # add a configuration file to save the hyperparameters in the experiment directory
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        for key, value in hparams.items():
            f.write(f'{key}: {value}\n')
    
    # Create the datasets and dataloaders
    dataset = CustomDataset(images_paths, hparams=hparams, transform=transforms.Compose([transforms.Normalize(mean=[hparams['mean']], std=[hparams['std']])]), device=device)
    #dataset = CustomDataset(images_paths, hparams=hparams, transform=None, device=device)
    
    # random split of Dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=True)
    
    plot_images(train_loader, experiment_dir, num_samples = 10)
    
    '''mean, std = calculate_mean_std_from_batches(train_loader, num_batches=50)
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        f.write(f'mean: {mean}\n')
        f.write(f'std: {std}\n')
    print('mean: ', mean)
    print('std: ',std)'''
    
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
    
    # add model configuration to the config file
    with open(os.path.join(experiment_dir, 'config.txt'), 'a') as f:
        f.write(f'Model: {model}\n')
        
    # add the number of parameters to the config file
    with open(os.path.join(experiment_dir, 'config.txt'), 'a') as f:
        f.write(f'Number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    model.to(device)
    
    # Check if weights file exists
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Define the loss function and optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, hparams['n_epochs'], device, log_dir=experiment_dir, patience = hparams['patience'], alpha=hparams['alpha'])
    
    # create the loss plot from the csv file and save it
    save_loss_plot(experiment_dir)
    
    # calculate metrics on the validation set and save them in a txt file
    evaluate_model(model, val_loader,best_model_path, experiment_dir, device)
    
    # plot results
    plot_results(model, val_loader, experiment_dir, device, num_samples=10)
    
    
    
if __name__ == "__main__":
    main()