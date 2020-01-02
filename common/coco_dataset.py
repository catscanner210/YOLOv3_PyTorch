import os
import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset

from common import data_transforms
# import data_transforms

class COCODataset(Dataset):
    def __init__(self, list_path, img_size, is_training, is_debug=False):
        self.img_files = []
        self.label_files = []
        for path in open(list_path, 'r'):
            path = '../data/coco' + path
            label_path = path.replace('images', 'labels').replace('.png', '.txt').replace(
                '.jpg', '.txt').strip()
            if os.path.isfile(label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)
            else:
                print("no image")
                logging.info("no label found. skip it: {}".format(path))
        logging.info("Total images: {}".format(len(self.img_files)))
        self.img_size = img_size  # (w, h)
        self.max_objects = 200
        self.is_debug = is_debug
        # print(len(self.label_files))
        #  transforms and augmentation
        #  the order cannot be switched cause boundingbox is augmented with the augmentation
        self.transforms = data_transforms.Compose()
        self.transforms.add(data_transforms.KeepAspect())
        self.transforms.add(data_transforms.ResizeImage(self.img_size))
        if is_training:
            self.transforms.add(data_transforms.ImageBaseAug())
        self.transforms.add(data_transforms.ToTensor(self.max_objects, self.is_debug))

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # img_path = img_path
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        # label_path = label_path 
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            logging.info("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)

        sample = {'image': img, 'label': labels}
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample

    def __len__(self):
        return len(self.img_files)


#  use for test dataloader
if __name__ == "__main__":
    dataloader = torch.utils.data.DataLoader(COCODataset("../data/coco/trainvalno5k.part",
                                                         (416, 416), True, is_debug=True),
                                             batch_size=16,
                                             shuffle=True, num_workers=1, pin_memory=True)
    for step, sample in enumerate(dataloader):
        # print(sample)
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # print(image.shape)
            cv2.imwrite("step{}_{}.jpg".format(step, i), image)
            # print(i)
        # print("step:{}".format(step))
        break # only one batch
