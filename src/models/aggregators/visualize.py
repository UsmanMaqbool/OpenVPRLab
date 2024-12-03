"""Visualization Utils"""
from PIL import Image
import torch
from torchvision import transforms
import os
import cv2
import numpy as np
__all__ = ['get_color_pallete']


def get_color_pallete(npimg, dataset='citys'):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    # recovery boundary
    if dataset in ('pascal_voc', 'pascal_aug'):
        npimg[npimg == -1] = 255
    # put colormap
    if dataset == 'ade20k':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(adepallete)
        return out_img
    elif dataset == 'citys':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cityspallete)
        return out_img
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(vocpallete)
    return out_img


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


vocpallete = _getvocpallete(256)

adepallete = [
    0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204,
    5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82,
    143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255,
    7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6,
    10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255,
    20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15,
    20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255,
    31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163,
    0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255,
    0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0,
    31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0,
    194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255,
    0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255,
    0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0,
    163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0,
    10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0,
    255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0,
    133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255]

cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]

def save_image(tensor_image, file_name):
    """
    Save a PyTorch tensor as an image file.

    Args:
    tensor_image (torch.Tensor): The image tensor to save.
    file_name (str): The name of the file to save the image to.
    # Example usage:
    # save_image(batch[0], 'output_image.png')
    """
    # Define the inverse normalize transform
    # inv_normalize = transforms.Normalize(
    #     mean=[-123.5, -116.75, -103.95],
    #     std=[255, 255, 255]
    # )

    inv_normalize = transforms.Normalize(
        mean=[-2.11790393, -2.03571429, -1.80444444],
        std=[4.36681223, 4.48275862, 4.44444444]
    )

    # Move the tensor to CPU if it's on GPU
    tensor_image = tensor_image.cpu()

    # Apply the inverse normalization
    tensor_image = inv_normalize(tensor_image)

    # Clip the image to be in the range [0, 1]
    tensor_image = torch.clamp(tensor_image, 0, 1)

    # Convert from (C, H, W) to (H, W, C) and to numpy array
    image = tensor_image.permute(1, 2, 0).numpy()

    # Convert to a PIL image
    image_pil = Image.fromarray((image * 255).astype('uint8'))  # Multiply by 255 if the image is in [0, 1] range

    # Save the image
    image_pil.save(file_name)
    

def save_batch_images(batch, base_dir='visualization'):
    """
    Save a batch of PyTorch tensors as image files.

    Args:
    batch (torch.Tensor): The batch of image tensors to save.
    base_dir (str): The base directory to save the images to.
    """
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Loop through each image in the batch
    for i in range(batch.size(0)):
        # Create a directory for each image
        img_dir = os.path.join(base_dir, str(i))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        # Save the image to the corresponding directory
        save_image(batch[i], os.path.join(img_dir, 's-1-image.png'))

def save_batch_masks(pred_batch, file_name, base_dir='visualization'):
    """
    Save a batch of masks to image files.

    Args:
    pred_batch (torch.Tensor): The batch of predictions.
    base_dir (str): The base directory to save the masks to.
    """
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Loop through each prediction in the batch
    for i in range(pred_batch.size(0)):
        # Generate the mask for each prediction
        mask = get_color_pallete(pred_batch[i].cpu().numpy(), 'citys')
        
        # Create a directory for each mask
        img_dir = os.path.join(base_dir, str(i))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        # Save the mask to the corresponding directory
        mask.save(os.path.join(img_dir, file_name))

def save_image_with_heatmap(tensor_image, pre_l2, img_i, file_name='image_with_heatmap.png', patch_idx=None):
    """
    Save a PyTorch tensor as an image file with an overlaid heatmap.

    Args:
    tensor_image (torch.Tensor): The original image tensor to save.
    pre_l2 (torch.Tensor): The tensor containing activations to generate the heatmap.
    file_name (str): The name of the file to save the image to.
    """
    # Define the inverse normalization
    inv_normalize = transforms.Normalize(
        mean=[-123.5, -116.75, -103.95],
        std=[255, 255, 255]
    )

    # Move the tensor to CPU if it's on GPU
    tensor_image = tensor_image.cpu()

    # Apply the inverse normalization
    tensor_image = inv_normalize(tensor_image)

    # Clip the image to be in the range [0, 1]
    tensor_image = torch.clamp(tensor_image, 0, 1)

    # Convert from (C, H, W) to (H, W, C) and to numpy array
    image = tensor_image.permute(1, 2, 0).numpy()

    # Convert to a PIL image
    image_pil = Image.fromarray((image * 255).astype('uint8'))

    # Convert the PIL image to a format compatible with OpenCV
    img = np.array(image_pil)

    # Compute the mean activations from pre_l2
    mean_activations = torch.mean(pre_l2.squeeze(), dim=0).detach().cpu().numpy()

    # Normalize the activations
    mean_activations = (mean_activations - np.min(mean_activations)) / (np.max(mean_activations) - np.min(mean_activations))

    # Get the dimensions of the original image
    height, width, _ = img.shape

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(mean_activations, (width, height))

    # Convert heatmap to an 8-bit format and apply a color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    # Convert the original image to BGR (OpenCV format)
    original_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Overlay the heatmap on the original image
    overlayed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    ## to store the patches
    H, W, _ = overlayed_img.shape
    bb_x = [
        [0, 0, int(2 * W / 3), H], 
        [int(W / 3), 0, W, H], 
        [0, 0, W, int(2 * H / 3)], 
        [0, int(H / 3), W, H], 
        [int(W / 4), int(H / 4), int(3 * W / 4), int(3 * H / 4)]
    ]

    
    if patch_idx == None:
        # Save the resulting image
        directory = f'visualization/{img_i}'
        full_file_name = os.path.join(directory, file_name)
        # Save the resulting image
        cv2.imwrite(full_file_name, overlayed_img)
        print(f"Saved image to {full_file_name}")
    else:
        # Save the resulting image
        directory = f'visualization/{img_i}'
        full_file_name = os.path.join(directory, file_name)
        # Save the resulting image
        cv2.imwrite(full_file_name, overlayed_img[bb_x[patch_idx][1]:bb_x[patch_idx][3], bb_x[patch_idx][0]:bb_x[patch_idx][2], :])        
        print(f"Saved image to {full_file_name}")
    
def save_x_nodes_patches(x_nodes, img_i, patch_idx):
    """
    Save a single image patch from the x_nodes tensor.

    Args:
    x_nodes (torch.Tensor): The tensor containing the image patch to save.
    img_i (int or str): The index of the image, used for naming the directory.
    patch_idx (int): The index of the patch, used for naming the file.
    """
    # Define the directory and ensure it exists
    directory = os.path.join('visualization', str(img_i))
    os.makedirs(directory, exist_ok=True)

    # Define the filename for each patch
    patch_file_name = f'patch_{patch_idx}.png'  # Customize the naming pattern as needed
    
    # Save the image patch
    save_image(x_nodes, os.path.join(directory, patch_file_name))

    print(f"Saved patch {patch_idx} to {os.path.join(directory, patch_file_name)}")
