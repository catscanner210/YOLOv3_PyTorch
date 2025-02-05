import numpy as np
import cv2
import torch

#  pip install imgaug
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)


class ToTensor(object):
    def __init__(self, max_objects=50, is_debug=False):
        self.max_objects = max_objects
        self.is_debug = is_debug

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        if self.is_debug == False:
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
        if len(labels) == 0 :
            labels = np.zeros((1,5),np.float32)
        filled_labels = np.zeros((self.max_objects, 5), np.float32)
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        return {'image': torch.from_numpy(image.copy()), 'label': torch.from_numpy(filled_labels.copy())}

class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        image_new = np.pad(image, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = image_new.shape

        # Extract coordinates for unpadded + unscaled image
        x1 = w * (label[:, 1] - label[:, 3]/2)
        y1 = h * (label[:, 2] - label[:, 4]/2)
        x2 = w * (label[:, 1] + label[:, 3]/2)
        y2 = h * (label[:, 2] + label[:, 4]/2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        label[:, 1] = ((x1 + x2) / 2) / padded_w
        label[:, 2] = ((y1 + y2) / 2) / padded_h
        label[:, 3] *= w / padded_w
        label[:, 4] *= h / padded_h
        # print("keep aspect:{}".format(label.shape))
        return {'image': image_new, 'label': label}

class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = tuple(new_size) #  (w, h)
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, self.new_size, interpolation=self.interpolation)
        return {'image': image, 'label': label}

class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)),
                    # iaa.AverageBlur(k=(2, 3)),
                    # iaa.MedianBlur(k=(3, 5)),
                ]),
                iaa.SomeOf((0, 3),[
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                # Add gaussian noise to some images.
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # Add a value of -5 to 5 to each pixel.
                sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                # Change brightness of images (80-120% of original value).
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                # Improve or worsen the contrast of images.
                # sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
                sometimes(iaa.LinearContrast((0.5, 2.0),per_channel=True)),
                # Affine Transfermation
                sometimes(iaa.Affine(scale=(0.9, 1.2), #expend image from 0.9X to 1.2X 
                                     rotate=(-15, 15),
                                     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                     shear=(-16, 16)),
                                     ),
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.15))),
                # sometimes(iaa.Fliplr(0.5)),
                # iaa.Flipud(0.5),
                ],
                random_order=True
                )
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
        # self.seq = iaa.Sequential(
        #     [
        #     # apply the following augmenters to most images
        #     iaa.Fliplr(0.5), # horizontally flip 50% of all images
        #     iaa.Flipud(0.2), # vertically flip 20% of all images
        #     sometimes(iaa.Affine(
        #         scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
        #         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
        #         rotate=(-10, 10), # rotate by -45 to +45 degrees
        #         shear=(-5, 5), # shear by -16 to +16 degrees
        #         order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        #         cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        #         mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        #     )),
        #     # execute 0 to 5 of the following (less important) augmenters per image
        #     # don't execute all of them, as that would often be way too strong
        #     iaa.SomeOf((0, 5),
        #         [
        #         sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
        #         iaa.OneOf([
        #             iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
        #             iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
        #             iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
        #         ]),
        #         iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
        #         iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
        #         # search either for all edges or for directed edges,
        #         # blend the result with the original image using a blobby mask
        #         iaa.SimplexNoiseAlpha(iaa.OneOf([
        #             iaa.EdgeDetect(alpha=(0.5, 1.0)),
        #             iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
        #         ])),
        #         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
        #         iaa.OneOf([
        #             iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
        #             iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
        #         ]),
        #         iaa.Invert(0.01, per_channel=True), # invert color channels
        #         iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
        #         iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
        #         # either change the brightness of the whole image (sometimes
        #         # per channel) or change the brightness of subareas
        #         iaa.OneOf([
        #             iaa.Multiply((0.9, 1.1), per_channel=0.5),
        #             iaa.FrequencyNoiseAlpha(
        #                 exponent=(-1, 0),
        #                 first=iaa.Multiply((0.9, 1.1), per_channel=True),
        #                 second=iaa.LinearContrast((0.5, 2.0),per_channel=True),
        #                 # second=iaa.ContrastNormalization((0.9, 1.1)),
        #             )
        #         ]),
        #         sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
        #         # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
        #         sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        #         ],
        #         random_order=True
        #     )
        #     ],
        #     random_order=True
        # )
    # def __call__(self, sample):
    #     seq_det = self.seq.to_deterministic()
    #     image, label = sample['image'], sample['label']
    #     # print(label.shape)
    #     image = seq_det.augment_images([image])[0]
    #     return {'image': image, 'label': label}

    def __call__(self, sample):
        seq_det = self.seq.to_deterministic()
        image, label = sample['image'], sample['label']
        # print(sample)

        h, w, _ = image.shape
        
        x1 = w * (label[:, 1] - label[:, 3]/2)
        y1 = h * (label[:, 2] - label[:, 4]/2)
        x2 = w * (label[:, 1] + label[:, 3]/2)
        y2 = h * (label[:, 2] + label[:, 4]/2)

        list_of_bbox = []
        for i in range(len(x1)):
            list_of_bbox.append(BoundingBox(x1=x1[i],y1=y1[i],x2=x2[i],y2=y2[i]))
 
        # print("length of list_of_bbox:{}".format(len(list_of_bbox)))
        bbs = BoundingBoxesOnImage(list_of_bbox,shape=image.shape)
        image_aug,bbs_aug = seq_det(image=image,bounding_boxes=bbs)
        # bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        bbs_aug = bbs_aug.remove_out_of_image()
        
        label_new = []
        for i in range(len(bbs_aug.bounding_boxes)):
            after = bbs_aug.bounding_boxes[i]
            x1a = after.x1
            y1a = after.y1
            x2a = after.x2
            y2a = after.y2

            try:
                A = BoundingBox(x1=x1a,y1=y1a,x2=x2a,y2=y2a)
                B = BoundingBox(x1=0,y1=0,x2=w-1,y2=h-1)
                R = A.intersection(B).area/A.area
                # print("R:{}".format(R))
                if  R <= 0.5: #Delete bbox that more than 1/3 area is out of the image
                    continue
                if  R < 1.0 and R>0.5: #clip bbox inside the image
                    xx = np.clip([x1a,x2a],0,w-1)
                    yy = np.clip([y1a,y2a],0,h-1)
                    x1a = xx[0]
                    x2a = xx[1]
                    y1a = yy[0]
                    y2a = yy[1]

                x0 = ((x1a + x2a) / 2.0) / w
                y0 = ((y1a + y2a) / 2.0) / h
                w0 = (x2a - x1a) / w
                h0 = (y2a - y1a) / h
                labela = label[:,0][i]
                
                label_new.append([labela,x0,y0,w0,h0])
            except Exception as e:
                print(e)
            

        return {'image': image_aug, 'label': np.asarray(label_new)}
