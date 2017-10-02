# coding: utf-8

import cv2 as cv
import numpy as np
import os
import chainer
from chainercv.utils import read_image
import pickle

vehicle_classes = (
    "car",)  # This must be tuple!!

class COWC_dataset_processed(chainer.dataset.DatasetMixin):
    def __init__(self,split="train",datadir ="E:/work/vehicle_detection_dataset/cowc_processed" ):
        self.data_dir = datadir
        id_list_file = os.path.join(
            self.data_dir, "list/{0}.txt".format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.split = split

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.
        Args:
            i (int): The index of the example.
        Returns:
            tuple of an image and bounding boxes
        """
        id_ = self.ids[i]
        imgfile = os.path.join(self.data_dir,self.split,"{0}.png".format(id_))
        annotation_file = os.path.join(self.data_dir,self.split,"{0}.txt".format(id_))

        bbox =[]
        label = []

        img = read_image(imgfile, color=True)

        with open(annotation_file,"r") as annotations:
            line = annotations.readline()
            while (line):
                xmin, ymin, xmax, ymax = line.split(",")
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                bbox.append([ymin-1, xmin-1, ymax-1, xmax-1]) #obey the rule of chainercv
                label.append(vehicle_classes.index("car"))
                line = annotations.readline()

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        return img, bbox, label

class COWC_fmap_set(chainer.dataset.DatasetMixin):
    def __init__(self,datadir, split="train"):
        self.data_dir = datadir
        id_list_file = os.path.join(
            self.data_dir, "list/{0}.txt".format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.split = split

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        fmp_file = os.path.join(self.data_dir,self.split,"{0}.fmp".format(id_))
        with open(fmp_file,"rb") as f:
            fmap = pickle.load(f)
        return fmap

class Dataset_imgonly(chainer.dataset.DatasetMixin):
    def __init__(self,datadir, split="train"):
        self.data_dir = datadir
        id_list_file = os.path.join(
            self.data_dir, "list/{0}.txt".format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.split = split

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        imgfile = os.path.join(self.data_dir,self.split,"{0}.tif".format(id_))
        img = read_image(imgfile, color=True)
        return img


if __name__ == "__main__":
    # a = COWC_dataset_processed("validation")
    # print(a.ids)
    # print(type(a.ids))
    # img, bbox, label = a[7]
    # print((img, bbox, label))
    a = Dataset_imgonly("E:/work/vehicle_detection_dataset/test_out")
    example = a[1]
    pass