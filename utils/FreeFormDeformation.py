import deepali.core.functional
import torch
from deepali.spatial import Grid, ImageTransformer, StationaryVelocityFreeFormDeformation
from torch import nn
from perlin import rand_perlin_2d
import numpy as np


def next8(number: int):
    if number % 8 == 0:
        return number
    return number + 8 - number % 8


class DeformationLayer(nn.Module):
    def __init__(self, shape, random_df_creation_setting = 0, stride=10) -> None:
        super().__init__()
        self.shape = shape
        grid = Grid(size=shape)
        self.field = StationaryVelocityFreeFormDeformation(grid, stride=stride, params=self.params)  # type: ignore
        self.field.requires_grad_(False)
        self.transformer = ImageTransformer(self.field)
        self.transformer_inv = ImageTransformer(self.field.inverse(link=True))
        self.random_df_creation_setting = random_df_creation_setting

    def params(self, *args, **kargs):
        # print(args, kargs)
        return self._parm

    def new_deformation(self, device):
        shape = self.field.data_shape
        s = (next8(shape[-2]), next8(shape[-1]))

        noise_2d = []
        
        if self.random_df_creation_setting == 0:
            random_scale_01 = 1
            random_scale_02 = 1
            random_scale_03 = 1
        if self.random_df_creation_setting == 1:
            x = np.random.uniform(0, 2.0)
            random_scale_01 = x
            random_scale_02 = x
            random_scale_03 = x
        if self.random_df_creation_setting == 2:
            random_scale_01 = np.random.uniform(0, 2.0)
            random_scale_02 = np.random.uniform(0, 2.0)
            random_scale_03 = np.random.uniform(0, 2.0)
        
        for i in range(shape[-3]):
            noise_2d_i = rand_perlin_2d(s, (4, 4)) * 0.05 * random_scale_01
            noise_2d_i += rand_perlin_2d(s, (8, 8)) * 0.03 * random_scale_02
            noise_2d_i += rand_perlin_2d(s, (2, 2)) * 0.2 * random_scale_03
            noise_2d_i = noise_2d_i[: shape[-2], : shape[-1]]
            noise_2d.append(noise_2d_i)

        self._parm = torch.stack(noise_2d, 0).unsqueeze(0).to(device)
        self.field.condition_()

    def deform(self, i: torch.Tensor):
        if len(i) == 3:
            i = i.unsqueeze(0)
        return self.transformer.forward(i)

    def back_deform(self, i: torch.Tensor):
        if len(i) == 3:
            i = i.unsqueeze(0)
        return self.transformer_inv.forward(i)

    def get_gird(self, stride=16, device=None):
        high_res_grid = self.field.grid().resize(self.shape[-2:])
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

def load_nii(path_to_nifty,slice_idx):
    from TPTBox import NII
    #from nii_wrapper import NII
    #from BIDS import NII
    from torchvision import datasets, transforms
    nii = NII.load(path_to_nifty, False)
    img = torch.Tensor(nii.get_array()[slice_idx,...].astype(np.float32)).T
    
    return img

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

    #img = load_png("square_dots.png")
    img = load_mnist(7)

    def show(*img):
        img = [(i.detach().cpu() if isinstance(i, torch.Tensor) else torch.from_numpy(i)) for i in img]
        img = [i / i.max() for i in img]

        img = [i.unsqueeze(0) if len(i.shape) == 2 else i for i in img]
        img = [i if len(i.shape) == 3 else i.squeeze(0) for i in img]
        #print(img)
        np_img = torch.cat(img, dim=-1).numpy()

        plt.figure(figsize=(20, 6))
        print(np_img.shape)
        plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation="nearest", cmap="gray")
        plt.show()

    i = img.unsqueeze(0)
    shape = i.shape[-2:]

    deform_layer = DeformationLayer(shape)

    with torch.no_grad():
        deform_layer.new_deformation(device)
        out = deform_layer.deform(i)
        out2 = deform_layer.back_deform(out)
        show(img.squeeze(), out.squeeze(), out2.squeeze(), deform_layer.deform(deform_layer.get_gird()))