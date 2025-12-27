import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,  padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]


#TEAM COLOMBIA - usually neural data is noisy and incomplete so in order to simulate those conditions we apply termporal masking, which can also prevent overfitting.
#Temporal Masking = Randomly hiding short chunks of the neural signal along the time axis during training.
import torch

def temporal_mask_gpu(x, max_mask_frac=0.15):
    """
    GPU temporal masking for batched input.

    Args:
        x (Tensor): shape (B, T, F)
        max_mask_frac (float): max fraction of T to mask

    Returns:
        Tensor: masked x
    """
    if not x.is_cuda:
        raise RuntimeError("temporal_mask_gpu expects CUDA tensor")

    B, T, F = x.shape

    # Sample mask lengths per batch element
    mask_lens = (torch.rand(B, device=x.device) * max_mask_frac * T).long()

    # Sample start indices
    max_starts = torch.clamp(T - mask_lens, min=1)
    starts = (torch.rand(B, device=x.device) * max_starts).long()

    # Create time indices
    time_idx = torch.arange(T, device=x.device)[None, :]  # (1, T)

    # Mask condition: (B, T)
    mask = (time_idx >= starts[:, None]) & (time_idx < (starts + mask_lens)[:, None])

    # Apply mask
    x = x.masked_fill(mask[:, :, None], 0.0)

    return x
