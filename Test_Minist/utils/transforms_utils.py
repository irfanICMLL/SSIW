import cv2
import numpy as np
import random
import torch
from typing import Optional, Tuple

def get_imagenet_mean_std() -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """ See use here in Pytorch ImageNet script: 
        https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197
        Returns:
        -   mean: Tuple[float,float,float], 
        -   std: Tuple[float,float,float] = None
    """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std
            

def normalize_img(  input: torch.Tensor, 
                    mean: Tuple[float,float,float], 
                    std: Optional[Tuple[float,float,float]] = None):
    """ Pass in by reference Torch tensor, and normalize its values.
        Args:
        -   input: Torch tensor of shape (3,M,N), must be in this order, and
                of type float (necessary).
        -   mean: mean values for each RGB channel
        -   std: standard deviation values for each RGB channel
        Returns:
        -   None
    """
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)

            
def pad_to_crop_sz(
    image: np.ndarray,
    crop_h: int,
    crop_w: int,
    mean: Tuple[float,float,float]
    ) -> Tuple[np.ndarray,int,int]:
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            src=image,
            top=pad_h_half,
            bottom=pad_h - pad_h_half,
            left=pad_w_half,
            right=pad_w - pad_w_half,
            borderType=cv2.BORDER_CONSTANT,
            value=mean)
    return image, pad_h_half, pad_w_half


def resize_by_scaled_short_side(
    image: np.ndarray,
    base_size: int,
    scale: float) -> np.ndarray:
    """ Equivalent to ResizeShort(), but functional, instead of OOP paradigm, and w/ scale param.

	Args:
	    image: Numpy array of shape ()
	    scale: scaling factor for image

	Returns:
	    image_scaled:
    """
    h, w, _ = image.shape
    short_size = round(scale * base_size)
    new_h = short_size
    new_w = short_size
    # Preserve the aspect ratio
    if h > w:
        new_h = round(short_size / float(w) * h)
    else:
        new_w = round(short_size / float(h) * w)
    image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image_scaled
