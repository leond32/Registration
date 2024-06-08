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
from tqdm import tqdm



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
        img = torch.tensor(img).float()
        shape = img.shape
        original_image = img.unsqueeze(0)  # Add batch dimension
        original_image= original_image.to(self.device)
        

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
    
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} - Training', leave=True)
        for i, (images, deformation_field) in enumerate(train_progress):
            # Move data to the device
            images = images.float().to(device)
            deformation_field = deformation_field.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, deformation_field)
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()
            train_progress.set_postfix(avg_loss=f'{train_loss/(i+1):.4f}', current_loss=f'{loss.item():.4f}')

        # Compute the average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} - Validation', leave=True)
        with torch.no_grad():
            for i, (images, deformation_field) in enumerate(val_progress):
                # Move validation data to the device
                images = images.float().to(device)
                deformation_field = deformation_field.to(device)
                
                outputs = model(images)
                batch_loss = criterion(outputs, deformation_field).item()
                val_loss += batch_loss
                val_progress.set_postfix(avg_loss=f'{val_loss/(i+1):.4f}', current_loss=f'{batch_loss:.4f}')
            
            
            # Compute the average validation loss
            avg_val_loss = val_loss / len(val_loader)
            
            # Print training and validation losses to console
            print(f'Training Loss (Epoch {epoch+1}/{n_epochs}): {avg_train_loss:.4f}')
            print(f'Validation Loss (Epoch {epoch+1}/{n_epochs}): {avg_val_loss:.4f}')  

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
    #print(device)

    train_data_path = '/home/ubuntu/ADLM/data/high-resolution/afhq/train'
    val_data_path = '/home/ubuntu/ADLM/data/high-resolution/afhq/val'

    train_images_paths = get_image_paths(train_data_path)
    val_images_paths = get_image_paths(val_data_path)
    
    mean = 0.5
    std = 0.5
    
    train_dataset = CustomDataset(train_images_paths, transform=transforms.Compose([transforms.Normalize(mean=[mean], std=[std])]), device=device)
    val_dataset = CustomDataset(val_images_paths, transform=transforms.Compose([transforms.Normalize(mean=[mean], std=[std])]), device=device)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

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
    if os.path.isfile('/home/ubuntu/ADLM/model_weights/model_weights_afhq.pth'):
        model.load_state_dict(torch.load('/home/ubuntu/ADLM/model_weights/model_weights_afhq.pth', map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0016, weight_decay=1e-5)

    n_epochs = 5
    train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
    
    # save (update the number of epochs in name)
   
    torch.save(model.state_dict(), '/home/ubuntu/ADLM/model_weights/model_weights_afhq.pth')

if __name__ == "__main__":
    main()

            
    
    

            
    
    