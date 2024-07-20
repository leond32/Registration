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
import sys



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
    
def early_stopping(val_losses, patience=5):
    if len(val_losses) < patience:
        return False
    for i in range(1, patience+1):
        if val_losses[-i] < val_losses[-i-1]:
            return False
    return True

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, log_dir='cifar10_logs', patience=5):
    # Create directories if they don't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Initialize lists to hold the loss values
    train_losses = []
    val_losses = []

    # Open CSV file for logging
    csv_path = os.path.join(log_dir, 'losses_64_02.csv')
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
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model_64_02.pth'))
            print(f'Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.8f}')
        

        # Check for early stopping
        if early_stopping(val_losses, patience):
            print('Early stopping...')
            break          
            
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = '/vol/aimspace/projects/practical_SoSe24/registration_group/datasets/CIFAR-10'
    cifar10_dataset = CIFAR10(data_path, train=True, download=True, transform=None)
    images_cifar10 = cifar10_dataset.data

    images = np.mean(images_cifar10, axis=3)
    images = torch.from_numpy(images)

    random_fraction = 1
    random_indices = random.sample(range(len(images)), int(random_fraction * len(images)))
    images = images[random_indices]
    transform = transforms.Resize((64, 64))
    images = transform(images)

    mean, std = get_mean_std(images)

    shape = images[0].shape[-2:]
    dataset = CustomDataset(images, shape, transform=transforms.Compose([transforms.Normalize(mean=[mean], std=[std])]), device=device)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

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
    if os.path.isfile('/vol/aimspace/projects/practical_SoSe24/registration_group/scripts/launchers/cifar10_logs/best_model_64.pth'):
        model.load_state_dict(torch.load('/vol/aimspace/projects/practical_SoSe24/registration_group/scripts/launchers/cifar10_logs/best_model_64.pth', map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)

    n_epochs = 20
    train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, log_dir='cifar10_logs')

    torch.save(model.state_dict(), '/vol/aimspace/projects/practical_SoSe24/registration_group/model_weights/model_weights_20_epochs_64_02.pth')

if __name__ == "__main__":
    main()
