import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from FreeFormDeformation import DeformationLayer
from deepali.core import functional as U
from tqdm import tqdm
import random
from diffusion_unet import Unet
from torch import nn, optim
import os
import cv2
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import torchvision.transforms.functional as F
import sys
from torch.utils.tensorboard import SummaryWriter


def get_mean_std(images):
    mean = torch.mean(images.float())
    std = torch.std(images.float())
    return mean.item(), std.item()



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
    


def early_stopping(val_losses, patience=5):
    if len(val_losses) < patience:
        return False
    for i in range(1, patience+1):
        if val_losses[-i] < val_losses[-i-1]:
            return False
    return True

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, log_dir='afhq_logs', patience=5):
    # Create directories if they don't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Initialize lists to hold the loss values
    train_losses = []
    val_losses = []

    # Open CSV file for logging
    csv_path = os.path.join(log_dir, 'losses_128.csv')
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
            loss = criterion(outputs, deformation_field)
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
                batch_loss = criterion(outputs, deformation_field).item()
                val_loss += batch_loss
                
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
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model_128.pth'))
            print(f'Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.8f}')
        

        # Check for early stopping
        if early_stopping(val_losses, patience):
            print('Early stopping...')
            break          
    
def get_image_paths(root_dir): 
    
    image_paths = []
    for category in os.listdir(root_dir):
        category_dir = os.path.join(root_dir, category)
        if os.path.isdir(category_dir):
            for filename in os.listdir(category_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_paths.append(os.path.join(category_dir, filename))   
    return image_paths
            
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_path = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/Animal_Faces/afhq/train'
    val_data_path = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/Animal_Faces/afhq/val'

    train_images_paths = get_image_paths(train_data_path)
    val_images_paths = get_image_paths(val_data_path)
    
    mean = 113
    std = 61
    
    train_dataset = CustomDataset(train_images_paths, transform=transforms.Compose([transforms.Normalize(mean=[mean], std=[std])]), device=device)
    val_dataset = CustomDataset(val_images_paths, transform=transforms.Compose([transforms.Normalize(mean=[mean], std=[std])]), device=device)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    model = Unet(
        dim=8,
        init_dim=None,
        out_dim=2,
        dim_mults=(1, 2, 4, 8, 16),
        channels=2,
        resnet_block_groups=8,
        learned_variance=False,
        conditional_dimensions=0,
        patch_size=1,
        attention_layer=None
    )

    model.to(device)
    # Check if weights file exists
    if os.path.isfile('/vol/aimspace/projects/practical_SoSe24/registration_group/model_weights/model_weights_afhq_128.pth'):
        model.load_state_dict(torch.load('/vol/aimspace/projects/practical_SoSe24/registration_group/model_weights/model_weights_afhq_128.pth', map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

    n_epochs = 30
    train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, log_dir='afhq_logs', patience = 5)
    
    # save (update the number of epochs in name)
   
    torch.save(model.state_dict(), '/vol/aimspace/projects/practical_SoSe24/registration_group/model_weights/model_weights_afhq_128.pth')

if __name__ == "__main__":
    main()

            
    
    

            
    
    