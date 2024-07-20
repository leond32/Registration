import deepali.core.functional
import torch
from deepali.spatial import Grid, ImageTransformer, StationaryVelocityFreeFormDeformation
from torch import nn
from perlin import rand_perlin_3d
import numpy as np
import torch.nn.functional as F


def next8(number: int):
    if number % 8 == 0:
        return number
    return number + 8 - number % 8

def next16(number: int):
    if number % 16 == 0:
        return number
    return number + 16 - number % 16

def smooth_field(field, kernel_size=5):
    padding = kernel_size // 2
    if field.ndim == 4:
        field = field.unsqueeze(0)  # Add batch dimension if missing
    elif field.ndim == 3:
        field = field.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions if missing
    smooth_field = F.avg_pool3d(field, kernel_size=kernel_size, stride=1, padding=padding)
    return smooth_field.squeeze(0)  # Remove the added dimension if needed


class DeformationLayer(nn.Module):
    def __init__(self, shape, fixed_img_DF=False, random_df_creation_setting = 0, stride=10) -> None:
        super().__init__()
        self.shape = shape
        grid = Grid(size=shape)
        self.field = StationaryVelocityFreeFormDeformation(grid, stride=stride, params=self.params)  # type: ignore
        self.field.requires_grad_(False)
        self.transformer = ImageTransformer(self.field)
        self.transformer_inv = ImageTransformer(self.field.inverse(link=True))
        self.random_df_creation_setting = random_df_creation_setting
        self.fixed_img_DF = fixed_img_DF

    def params(self, *args, **kargs):
        # print(args, kargs)
        return self._parm
    

    def new_deformation(self, device):
        shape = self.field.data_shape
        #print(shape)
        s = (next16(shape[-3]), next16(shape[-2]), next16(shape[-1]))

        noise_3d = []
        
        if self.random_df_creation_setting == 0:
            random_scale_01 = 1
            random_scale_02 = 1
            random_scale_03 = 1
        if self.random_df_creation_setting == 1:
            x = np.random.uniform(0, 1.0)
            random_scale_01 = x
            random_scale_02 = x
            random_scale_03 = x
        if self.random_df_creation_setting == 2:
            random_scale_01 = np.random.uniform(0, 4) #4
            random_scale_02 = np.random.uniform(0, 3) #2
            random_scale_03 = np.random.uniform(0, 2) #1.5
        
        if self.fixed_img_DF:
            # If fixed_img_DF is True, create a small deformation that is applied to the fixed and moving image (used for data augmentation)
            random_scale = np.random.uniform(0, 1)
            for i in range(shape[-4]):
                noise_3d_i = rand_perlin_3d(s, (8, 8, 8)) * 0.05 * random_scale
                noise_3d_i += rand_perlin_3d(s, (16, 16, 16)) * 0.03 * random_scale
                noise_3d_i += rand_perlin_3d(s, (4, 4, 4)) * 0.2 * random_scale
                noise_3d_i = noise_3d_i[: shape[-3], : shape[-2], : shape[-1]]
                noise_3d.append(noise_3d_i)
        else:
            for i in range(shape[-4]):
                noise_3d_i = rand_perlin_3d(s, (8, 8, 8)) * 0.05 * random_scale_01
                noise_3d_i += rand_perlin_3d(s, (16, 16, 16)) * 0.03 * random_scale_02
                noise_3d_i += rand_perlin_3d(s, (4, 4, 4)) * 0.2 * random_scale_03
                noise_3d_i = noise_3d_i[: shape[-3], : shape[-2], : shape[-1]]
                noise_3d.append(noise_3d_i)
    
        
        self._parm = torch.stack(noise_3d, 0).unsqueeze(0).to(device)
        self.field.condition_()

    def deform(self, i: torch.Tensor):
        if len(i) == 4:
            i = i.unsqueeze(0)
    
        return self.transformer.forward(i)
    

    def back_deform(self, i: torch.Tensor):
        if len(i) == 4:
            i = i.unsqueeze(0)
        return self.transformer_inv.forward(i)

    def get_gird(self, stride=16, device=None):
        high_res_grid = self.field.grid().resize(self.shape[-3:])
        return deepali.core.functional.grid_image(high_res_grid, num=1, stride=stride, inverted=True, device=device)
    
    def get_deformation_field(self):
            self.field.update()
            # Access the displacement field buffer 'u'
            displacement_field = self.field.u  # Assuming it has shape [2, Height, Width]
            return displacement_field

def load_png(name):
    import torch
    import sys
    from PIL import Image
    import os

    sys.path.append("/res")

    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, '../images/',name)

    image = Image.open(image_path)

    # Convert PIL image to numpy array and transpose it
    image_array = np.array(image).astype(np.float32)

    # Convert numpy array to PyTorch tensor
    img = torch.tensor(image_array)

    # Extract the red channel to create a grayscale image
    red_channel = image_array[:, :, 0]  # Only take the red channel

    # Convert the red channel numpy array to PyTorch tensor
    img = torch.tensor(red_channel).unsqueeze(0)

    return img

def load_mnist(index):
    import torchvision.datasets as datasets
    from torchvision import transforms

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST training dataset
    trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Get one batch of training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Select one image from the batch
    image_index = 7
    img = images[image_index]

    return img

def load_nii(path_to_nifty):
    from TPTBox import NII
    nii = NII.load(path_to_nifty, False)
    t_tensor = torch.Tensor(nii.get_array().astype(np.float32))

    return t_tensor

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision
    from torchvision import transforms
    from PIL import Image
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    img = load_nii("D:\\Dokumente\\03_RCI\\practical\\Folder_structure\\rawdata_normalized\\sub-0003\\T2w\\sub-0003_T2w.nii.gz")

    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('Use scroll wheel to navigate images')

            self.X = X
            self.slices, self.rows, self.cols = X.shape
            self.ind = self.slices // 2

            self.im = ax.imshow(self.X[self.ind, :, :], cmap="gray")
            self.update()

        def onscroll(self, event):
            #print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            self.im.set_data(self.X[self.ind, :, :])
            self.ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()

    def plot3d(image):
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, image)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

        
    def show3d(*images, axis=0, index=0):
        """
        Display slices of multiple 3D images.

        Parameters:
        images (list of torch.Tensor or np.ndarray): The 3D images to be displayed.
        axis (int): The axis along which to take the slice (0 for axial, 1 for coronal, 2 for sagittal).
        index (int): The index of the slice along the specified axis.
        """
        # Process each image
        slices = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            
            if len(img.shape) != 3:
                #print(img.shape)
                raise ValueError("Each input image must be a 3D tensor or array.")
            
            # Normalize the image
            img = img / img.max()
            
            # Select the slice
            if axis == 0:
                slice_img = img[index, :, :]
            elif axis == 1:
                slice_img = img[:, index, :]
            elif axis == 2:
                slice_img = img[:, :, index]
            else:
                raise ValueError("Axis must be 0 (axial), 1 (coronal), or 2 (sagittal).")
            
            slices.append(slice_img)
        
        # Plot all slices
        num_slices = len(slices)
        plt.figure(figsize=(6 * num_slices, 6))
        for i, slice_img in enumerate(slices):
            plt.subplot(1, num_slices, i + 1)
            plt.imshow(slice_img, cmap="gray", interpolation="nearest")
            plt.title(f"Slice {index} along axis {axis}, Image {i+1}")
        plt.show()
    
    i = img.unsqueeze(0)
    j = img.squeeze(0)
    shape = j.T.shape[-3:]

    deform_layer = DeformationLayer(shape)
    deform_layer = DeformationLayer(shape, fixed_img_DF=False, random_df_creation_setting=2)
    with torch.no_grad():
        deform_layer.new_deformation(device)
        out = deform_layer.deform(i)
        out2 = deform_layer.back_deform(out)
        #show3d(img.squeeze(), out.squeeze(), out2.squeeze(), deform_layer.deform(deform_layer.get_gird()).squeeze(), axis=0, index=8)
        #plot3d(img.squeeze().numpy())
        plot3d(out.squeeze().numpy())
            