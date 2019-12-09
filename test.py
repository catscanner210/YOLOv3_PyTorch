import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

# image = torch.randn((8,3,416,416))
# print(image.size())

# output = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=2,stride=2)(image)
# print(output.size())

img_bgr = cv.imread('step0_0.jpg')
# img_rgb = img_bgr[:,:,[2,1,0]]
img_rgb = cv.cvtColor(img_bgr,cv.COLOR_BGR2RGB)
img_rgb = cv.resize(img_rgb,(416,416))
img_rgb_chw = np.transpose(img_rgb,(2,0,1))

img = torch.from_numpy(img_rgb_chw)
img_torch = img.float().div(255).unsqueeze(0)

print(img_torch.size())

conv1_out = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=2,stride=2)(img_torch)

print(conv1_out[:,:3,:,:].size())

imshow(conv1_out[:,:3,:,:])
imshow(conv1_out[:,3:6,:,:])
imshow(conv1_out[:,6:9,:,:])


conv2_out = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=2,stride=2)(img_torch)

print(conv2_out.size())

new_conv = conv2_out + conv1_out

print(new_conv.size())