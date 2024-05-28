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



def get_mean_std(images):
    mean = torch.mean(images.float())
    std = torch.std(images.float())
    return mean.item(), std.item()



class CustomDataset(Dataset):
    def __init__(self, images, shape,transform=None, device = "cpu"):
        """
        Args:
            images (torch.Tensor): The tensor containing image data.
            shape: The shape of one image in the dataset.
            mean (float): The mean value for normalization.
            std (float): The standard deviation for normalization.
            transform (bool): Whether to apply the transformation.
        """
        self.images = images
        self.shape = shape
        self.transform = transform
        self.device = device
    
    def __len__(self):
        return len(self.images)
    
    def build_deformation_layer(self, device):
        """
        Build and return a new deformation layer for each call to __getitem__.
        This method returns the created deformation layer.
        """
        deformation_layer = DeformationLayer(self.shape)
        deformation_layer.new_deformation(device=device)
        return deformation_layer

    def __getitem__(self, idx):
        # Fetch the original image
        original_image = self.images[idx].unsqueeze(0)  # Add batch dimension
        original_image = original_image.to(self.device)

        # Build a new deformation layer for the current image
        deformation_layer = self.build_deformation_layer(self.device).to(self.device)

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
        # Loop through training batches
        for i, (images, deformation_field) in enumerate(train_loader):
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

        # Compute the average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        # Loop through validation batches
        with torch.no_grad():
            for i, (images, deformation_field) in enumerate(val_loader):
                # Move validation data to the device
                images = images.float().to(device)
                deformation_field = deformation_field.to(device)
                
                outputs = model(images)
                batch_loss = criterion(outputs, deformation_field).item()
                val_loss += batch_loss
            
            # Compute the average validation loss
            avg_val_loss = val_loss / len(val_loader)
            
            # Print training and validation losses to console
            print(f'Training Loss (Epoch {epoch+1}/{n_epochs}): {avg_train_loss:.4f}')
            print(f'Validation Loss (Epoch {epoch+1}/{n_epochs}): {avg_val_loss:.4f}')  

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
    return images         
            
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/AnimalFaces/afhq'
    train_images = load_images(os.path.join(data_path, 'train/cat')) + load_images(os.path.join(data_path, 'train/dog')) + load_images(os.path.join(data_path, 'train/wild'))
    val_images = load_images(os.path.join(data_path, 'val/cat')) + load_images(os.path.join(data_path, 'val/dog')) + load_images(os.path.join(data_path, 'val/wild'))

    train_images = np.mean(train_images, axis=3)
    train_images = torch.from_numpy(train_images)
    
    val_images = np.mean(val_images, axis=3)
    val_images = torch.from_numpy(val_images)


    mean, std = get_mean_std(train_images + val_images)

    shape = train_images[0].shape[-2:]
    train_dataset = CustomDataset(train_images, shape, transform=transforms.Compose([transforms.Normalize(mean=[mean], std=[std])]), device=device)
    val_dataset = CustomDataset(val_images, shape, transform=transforms.Compose([transforms.Normalize(mean=[mean], std=[std])]), device=device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

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
    if os.path.isfile('/vol/aimspace/projects/practical_SoSe24/registration_group/model_weights/model_weights_afhq.pth'):
        model.load_state_dict(torch.load('/vol/aimspace/projects/practical_SoSe24/registration_group/model_weights/model_weights_afhq.pth', map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)

    n_epochs = 10
    train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
    
    # save (update the number of epochs in name)
   
    torch.save(model.state_dict(), '/vol/aimspace/projects/practical_SoSe24/registration_group/model_weights/model_weights_afhq.pth')

if __name__ == "__main__":
    main()

            
    
    