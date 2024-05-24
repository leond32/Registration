import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from FreeformTry import DeformationLayer
from deepali.core import functional as U
from tqdm import tqdm
import random
from diffusion_unet import Unet
from torch import nn, optim



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
    
    
    
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs):
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        # Initialize tqdm progress bar for the training batches
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training Epoch {epoch+1}/{n_epochs}')
        for i, (images, deformation_field) in train_bar:
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

            # Update the progress bar with the latest loss information
            train_bar.set_postfix(loss=loss.item())
        
        # Compute the average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        # Initialize tqdm progress bar for the validation batches
        val_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation')
        with torch.no_grad():
            for i, (images, deformation_field) in val_bar:
                # Move validation data to the device
                images = images.float().to(device)
                deformation_field = deformation_field.to(device)
                
                outputs = model(images)
                batch_loss = criterion(outputs, deformation_field).item()
                val_loss += batch_loss
                
                # Update the progress bar for validation
                val_bar.set_postfix(val_loss=batch_loss)
            
            # Compute the average validation loss
            avg_val_loss = val_loss / len(val_loader)
            print(f'Validation Loss: {avg_val_loss}')
            
            
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
    transform = transforms.Resize((32, 32))
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
    if os.path.isfile('model_weights.pth'):
        model.load_state_dict(torch.load('model_weights.pth', map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)

    n_epochs = 50
    train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs)

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    main()
            
    
    