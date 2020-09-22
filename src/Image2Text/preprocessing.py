import torch.nn.functional as F
import torch
from skimage import transform

import config

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
        image = (image > self.threshold).type('torch.FloatTensor')
        return image

class TransposeImage(object):
    def __init__(self):
        return

    def __call__(self, image):
        image = torch.transpose(image, 1, 2)
        return image
