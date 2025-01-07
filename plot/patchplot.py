import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_patch_image(
    image: Image.Image,
    img_size: int,
    patch_size: int    
):
    """Plot one image as many patches. img_size must be compatible with patch_size.

    Args:
        image (Image.Image): input image
        img_size (int): image size
        patch_size (int): patch size in pixels
    """
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    
    num_patches = img_size / patch_size
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"
    print(f"Number of patches per row: {num_patches}\
        \nPatch size: {patch_size} x {patch_size} pixels\
        \nTotal patches: {num_patches*num_patches}")
    
    # Create a series of subplots
    fig, axs = plt.subplots(
        nrows=img_size // patch_size,
        ncols=img_size // patch_size,
        figsize=(num_patches, num_patches),
        sharex=True,
        sharey=True
    )
    
    for i, patch_height in enumerate(range(0, img_size, patch_size)):
        for j, patch_width in enumerate(range(0, img_size, patch_size)):
            axs[i, j].imshow(image_array[patch_height:patch_height+patch_size, patch_width:patch_width+patch_size, :])
            axs[i, j].set_ylabel(
                i+1,
                rotation="horizontal",
                horizontalalignment="right",
                verticalalignment="center"
            )
            axs[i, j].set_xlabel(j+1)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            axs[i,j].label_outer()
    plt.show()
        
image = Image.open("image.jpeg")


plt.imshow(image)
plt.axis(False)

plot_patch_image(image, 600, 60)