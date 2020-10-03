import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from skimage import transform

from Image2Text import config

class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample[0]

        h, w = image.shape[:2]
        if (h / w) > (self.output_size[0] / self.output_size[1]):
            req_w = (h * self.output_size[1]) / self.output_size[0]
            image = F.pad(image, (0, int(req_w) - w, 0, 0), value=239, mode="constant")
        elif (h / w) < (self.output_size[0] / self.output_size[1]):
            req_h = (w * self.output_size[0]) / self.output_size[1]
            image = F.pad(image, (0, 0, int((req_h - h) // 2), int((req_h - h) // 2)), value=239, mode="constant")
        
        new_h, new_w = self.output_size
        image = image.squeeze(0)
        image = torch.tensor(transform.resize(image, (new_h, new_w)))
        
        return image.view(1, config.IMAGE_H, config.IMAGE_W)
        
class GreyscaleToBlackAndWhite(object):
    def __init__(self):
        self.threshold = config.THRESHOLD

    def __call__(self, image):
        image = (image <= self.threshold).type('torch.FloatTensor')
        return image

class TransposeImage(object):
    def __init__(self):
        return

    def __call__(self, image):
        image = torch.transpose(image, 1, 2)
        return image

class GaussianFiltering(object):
    def __init__(self, channels=1, kernel_size=7, sigma=3):
        # Set these to whatever you want for your gaussian filter
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        self.x_cord = torch.arange(self.kernel_size)
        self.x_grid = self.x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        self.y_grid = self.x_grid.t()
        self.xy_grid = torch.stack([self.x_grid, self.y_grid], dim=-1)

        self.mean = (self.kernel_size - 1) / 2.
        self.variance = self.sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        self.gaussian_kernel = (1./(2. * math.pi * self.variance)) *\
                        torch.exp(
                            -torch.sum((self.xy_grid - self.mean)**2., dim=-1) /\
                            (2*self.variance)
                        )
        # Make sure sum of values in gaussian kernel equals 1.
        self.gaussian_kernel = self.gaussian_kernel / torch.sum(self.gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        self.gaussian_kernel = self.gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        self.gaussian_kernel = self.gaussian_kernel.repeat(self.channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.channels, bias=False)

        self.gaussian_filter.weight.data = self.gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def __call__(self, image):
        return self.gaussian_filter(image.unsqueeze(0)).squeeze(0)

class AverageFiltering(object):
    def __init__(self, channels=1, kernel_size=5):
        # Set these to whatever you want for your gaussian filter
        self.kernel_size = kernel_size
        self.channels = channels

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        self.averaging_kernel = torch.ones((self.kernel_size, self.kernel_size))
        self.averaging_kernel = self.averaging_kernel / torch.sum(self.averaging_kernel)

        # Reshape to 2d depthwise convolutional weight
        self.averaging_kernel = self.averaging_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        self.averaging_kernel = self.averaging_kernel.repeat(self.channels, 1, 1, 1)

        self.averaging_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.channels, bias=False)

        self.averaging_filter.weight.data = self.averaging_kernel
        self.averaging_filter.weight.requires_grad = False

    def __call__(self, image):
        return self.averaging_filter(image.unsqueeze(0)).squeeze(0)

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x.unsqueeze(0), self._padding(x.unsqueeze(0)), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        x = x.squeeze(0)
        return x

class NumpyToTensor(object):
    def __init__(self):
        return
    def __call__(self, image):
        return torch.Tensor(image.float())