import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os,glob,shutil

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    print(image.ndimension())
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def imsave(tensor, root_path,file_name):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    # print(image.ndimension())
    image = transforms.ToPILImage()(image)
    image.save(os.path.join(root_path,file_name))


def read_img_cv2(img_path):
    # image = torch.randn((8,3,416,416))
    # print(image.size())
    # output = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=2,stride=2)(image)
    # print(output.size())
    img_bgr = cv.imread(img_path)
    # img_rgb = img_bgr[:,:,[2,1,0]]
    img_rgb = cv.cvtColor(img_bgr,cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb,(416,416))
    img_rgb_chw = np.transpose(img_rgb,(2,0,1))
    img = torch.from_numpy(img_rgb_chw)
    img_torch = img.float().div(255).unsqueeze(0)
    # print(img_torch.size())
    return img_torch

def read_img(img_path):
    input_image = Image.open(img_path)
    preprocess = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch

def my_conv(img_torch):
    conv1_out = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=2,stride=2)(img_torch)
    print(conv1_out[:,:3,:,:].size())
    imshow(conv1_out[:,:3,:,:])
    imshow(conv1_out[:,3:6,:,:])
    imshow(conv1_out[:,6:9,:,:])
    conv2_out = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=2,stride=2)(img_torch)
    print(conv2_out.size())
    new_conv = conv2_out + conv1_out
    print(new_conv.size())

def export_onnx(model,model_name,w=416,h=416):
    import torch.onnx
    # 按照输入格式，设计随机输入
    dummy_input =torch.randn(1, 3, w, h)
    # 导出模型
    torch.onnx.export(net,dummy_input, model_name, verbose=True)

def concat_tensor(tensors_tuple):
    image1 = read_img_cv2('')
    image2 = read_img_cv2('')
    batch = torch.cat((image1,image2),0)
    print(batch.size())
    output = net(batch)
    print(output.size())

def summary_model(model):
    from torchsummary import summary
    summary(model, (3, 416, 416))

def softmax_output(output):
    return torch.nn.functional.softmax(output[0], dim=0)

def get_layer_index(net,layer_name):
    return list(dict(net.named_children()).keys()).index(layer_name)

class MyResNet50(nn.Module):
    def __init__(self, original_model,layer_name):
        super(MyResNet50, self).__init__()
        # self.features = nn.Sequential(*list(original_model.children())[:-2])
        index = get_layer_index(original_model,layer_name)
        self.features = nn.Sequential(*list(original_model.children())[:index+1])

    def forward(self, x):
        x = self.features(x)
        return x

def print_each_layer(net):
    for layer in list(net.children()):
        print(layer)

# img_path = '2.jpeg'
# output = sub_net(read_img(img_path))
# print(output.size())

def featuremaps_of_one_image(output,img_path,layer):
    root = '../featuremaps/{}_{}'.format(layer,img_path)
    print("output folder:{}".format(root))
    if(os.path.exists(root)):
        # print("root:{}  exist!".format(root))
        shutil.rmtree(root)
    os.makedirs(root)
    feature_layers = output.size()[1]
    print('featurelayers:{}'.format(feature_layers))
    for i in range(feature_layers):
        imsave(output[:,i:i+1,:,:],root,'{}.jpg'.format(i))

def featuremaps_of_folder(net,folder_path,layer):
    for img in glob.glob(folder_path):
        # print(img)
        output = net(read_img(img).to(device))
        featuremaps_of_one_image(output,os.path.basename(img),layer)

def get_multilayer_features(layers,resnet):
    for layer in layers:
        net = MyResNet50(resnet,layer).to(device)
        net.eval()
        featuremaps_of_folder(net,'../inputs/*',layer)


layers = ['layer1']
resnet = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=True)
resnet.eval()
# print(resnet)
get_multilayer_features(layers,resnet)

