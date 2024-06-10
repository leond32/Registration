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

def get_mean_std(images):
    mean = torch.mean(images.float())
    std = torch.std(images.float())
    return mean.item(), std.item()

############################################################################################################

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None, device = "cpu"):
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
    
    def __len__(self):
        return len(self.image_paths)
    
    def build_deformation_layer(self, shape, device):
        """
        Build and return a new deformation layer for each call to __getitem__.
        This method returns the created deformation layer.
        """
        deformation_layer = DeformationLayer(shape)
        deformation_layer.new_deformation(device=device)
        return deformation_layer

    def __getitem__(self, idx):
        # Fetch the original image
        image_path = self.image_paths[idx]
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        transform = transforms.Resize((128,128))
        img = transform(img)
        img = F.pil_to_tensor(img).float()
        shape = img.squeeze(0).shape
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
    for category in os.listdir(root_dir):
        category_dir = os.path.join(root_dir, category)
        if os.path.isdir(category_dir):
            for filename in os.listdir(category_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_paths.append(os.path.join(category_dir, filename))   
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

def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())

############################################################################################################

def build_box_plot(data, title, x_label, y_label, save_path):
    
    # Boxplot of SSIM, PSNR, MSE, L1
    plt.boxplot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.close()
    
############################################################################################################
def compute_metrics(model, val_loader, device):
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

def evaluate_model(model, val_loader, experiment_dir, device):
    # Compute similarity measures of image and redeformed image from validation data (SSIM, PSNR, MSE, L1)
    avg_ssim, avg_psnr, avg_mse, avg_l1, ssim_values, psnr_values, mse_values, l1_values = compute_metrics(model, val_loader, device)
    
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
    train_data_path = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/Animal_Faces/afhq/train'
    val_data_path = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/Animal_Faces/afhq/val'
    
    # Define the paths to save the logs and the best model	
    log_dir = '/vol/aimspace/projects/practical_SoSe24/registration_group/AFHQ_Experiments' # Change if you dont train on AFHQ
    experiment_name = 'Experiment_08' # Change this to a different name for each experiment 
    experiment_dir = os.path.join(log_dir, experiment_name)
    best_model_path = os.path.join(experiment_dir,'best_model.pth')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # Get the image paths for dynamic dataset creation    
    train_images_paths = get_image_paths(train_data_path)
    val_images_paths = get_image_paths(val_data_path)
    
    # Define the hyperparameters
    hparams = {
        'mean': 113,
        'std': 61,
        'n_epochs': 50,
        'batch_size': 16,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'patience': 5,
        'alpha': 0.00005
    }
    
    # add a configuration file to save the hyperparameters in the experiment directory
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        for key, value in hparams.items():
            f.write(f'{key}: {value}\n')
    
    # Create the datasets and dataloaders
    train_dataset = CustomDataset(train_images_paths, transform=transforms.Compose([transforms.Normalize(mean=[hparams['mean']], std=[hparams['std']])]), device=device)
    val_dataset = CustomDataset(val_images_paths, transform=transforms.Compose([transforms.Normalize(mean=[hparams['mean']], std=[hparams['std']])]), device=device)
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=True)

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

    model.to(device)
    
    # Check if weights file exists
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Define the loss function and optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, hparams['n_epochs'], device, log_dir=experiment_dir, patience = hparams['patience'], alpha=hparams['alpha'])
    '''
    # after training, save an image of the loss and calculate some metrics on the validatuion set
    if not os.path.exists(os.path.join(experiment_dir, 'images')):
        os.makedirs(os.path.join(experiment_dir, 'images'))
    if not os.path.exists(os.path.join(experiment_dir, 'metrics')):
        os.makedirs(os.path.join(experiment_dir, 'metrics'))
    '''   
    # create the loss plot from the csv file and save it
    save_loss_plot(experiment_dir)
    '''csv_path = os.path.join(experiment_dir, 'losses.csv')
    df = pd.read_csv(csv_path)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'images', 'loss_plot.png'))
    plt.close()'''
    
    # calculate metrics on the validation set and save them in a txt file
    evaluate_model(model, val_loader, experiment_dir, device)
    '''avg_ssim, avg_psnr, avg_mse, avg_l1, ssim_values, psnr_values, mse_values, l1_values = compute_metrics(model, val_loader, device)
    print('Metrics calculated')
    print(f'Average SSIM: {avg_ssim}, Average PSNR: {avg_psnr}, Average MSE: {avg_mse}, Average L1: {avg_l1}')
    with open(os.path.join(experiment_dir, 'metrics', 'metrics.txt'), 'w') as f:
        f.write(f'Average SSIM: {avg_ssim}\n')
        f.write(f'Average PSNR: {avg_psnr}\n')
        f.write(f'Average MSE: {avg_mse}\n')
        f.write(f'Average L1: {avg_l1}\n')'''
    
    
if __name__ == "__main__":
    main()

            
    
    

            
    
    