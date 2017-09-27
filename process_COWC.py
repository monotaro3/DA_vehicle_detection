# coding: utf-8

import cv2 as cv
import numpy as np
import os
import math
from utils import decode_mask2bbox

process_directories = []
# process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Potsdam_ISPRS")
# process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Selwyn_LINZ")
# process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Toronto_ISPRS")
# process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Utah_AGRC")

save_directory = ""

directories_list = "process_COWC_dirs.txt"

dir_lists = [dir_.strip() for dir_ in open(directories_list)]
for dir_ in dir_lists:
    record = dir_.split(",")
    if record[0] == "process" : process_directories.append(record[1])
    elif record[0] == "save": save_directory = record[1]
    else:
        try:
            raise ValueError("invalid directory specification")
        except ValueError as e:
            print(e)

print("process dirs:")
print(process_directories)

output_size = 300
windowsize = 50
adjust_margin = False
margin = 0
use_edge = False
scale = 0.5 # set None when not using, 0.5 -> halve the resolution

train_img_number = 0
test_img_number = 0
all_img_number = 0
train_imglist_text = os.path.join(save_directory,"list","train.txt")
test_imglist_text = os.path.join(save_directory,"list","validation.txt")

if not os.path.isdir(os.path.join(save_directory,"train")):
    os.makedirs(os.path.join(save_directory,"train"))
if not os.path.isdir(os.path.join(save_directory, "validation")):
    os.makedirs(os.path.join(save_directory, "validation"))
if not os.path.isdir(os.path.join(save_directory, "list")):
    os.makedirs(os.path.join(save_directory, "list"))



def scale_img_bbox(img, bbox, output_size):
    h,w,c = img.shape
    scale = output_size / h
    img_ = cv.resize(img,(output_size,output_size))
    bbox_ = []
    for b in bbox:
        xmin = math.floor(b[0] * scale)
        ymin = math.floor(b[1] * scale)
        xmax = math.floor(b[2] * scale)
        ymax = math.floor(b[3] * scale)
        b_ = [xmin,ymin,xmax,ymax]
        for i in range(len(b_)):
            if b_[i] == 0: b_[i] = 1
            if b_[i] > output_size: b_[i] = output_size
        bbox_.append(b_)
    return img_, bbox_

def make_img_cutouts(image_path,image_mask_path,save_directory,cutout_size,size_output,windowsize):
    image = cv.imread(image_path)
    image_mask = cv.imread(image_mask_path)
    height, width, channel = image.shape
    pointer = [0,0] #W,H
    #cutout_ = np.empty((cutout_size,cutout_size,3))
    #cutout_mask_ = np.empty((cutout_size, cutout_size, 3))
    write_flag = False
    global train_img_number
    global test_img_number
    global all_img_number
    global train_imglist_text
    global test_imglist_text
    global adjust_margin
    global margin
    if not adjust_margin:
        W_slot = int((width-margin)/(cutout_size-margin))
        H_slot = int((height-margin)/(cutout_size-margin))
        for H in range(H_slot+1):
            if H != H_slot or use_edge:
                for W in range(W_slot+1):
                    if H == H_slot:
                        if use_edge:
                            if W == W_slot:
                                cutout_ = image[-cutout_size:,-cutout_size:,:]
                                cutout_mask_ = image_mask[-cutout_size:, -cutout_size:, :]
                                write_flag = True
                            else:
                                cutout_ = image[-cutout_size:, (cutout_size-margin)*W:cutout_size*(W+1)+margin, :]
                                cutout_mask_ = image_mask[-cutout_size:, (cutout_size-margin)*W:cutout_size*(W+1)+margin, :]
                                write_flag = True
                    else:
                        if W == W_slot:
                            if use_edge:
                                cutout_ = image[(cutout_size-margin)*H:cutout_size*(H+1)+margin, -cutout_size:, :]
                                cutout_mask_ = image_mask[(cutout_size-margin)*H:cutout_size*(H+1)+margin, -cutout_size:, :]
                                write_flag = True
                        else:
                            cutout_ = image[(cutout_size-margin)*H:cutout_size*(H+1)+margin,  (cutout_size-margin)*W:cutout_size*(W+1)+margin, :]
                            cutout_mask_ = image_mask[(cutout_size-margin)*H:cutout_size*(H+1)+margin,  (cutout_size-margin)*W:cutout_size*(W+1)+margin, :]
                            write_flag = True
                    if write_flag:
                        write_flag = False
                        bbox = decode_mask2bbox(cutout_mask_,windowsize)
                        if len(bbox) > 0:
                            all_img_number += 1
                            if all_img_number % 4 != 0:
                                usage = "train"
                                train_img_number += 1
                                img_number = train_img_number
                            else:
                                usage = "validation"
                                test_img_number += 1
                                img_number = test_img_number
                            filename = '{0:010d}.png'.format(img_number)
                            filename_annotation = '{0:010d}.txt'.format(img_number)

                            if cutout_size != size_output:
                                cutout_, bbox = scale_img_bbox(cutout_, bbox, size_output)

                            cv.imwrite(os.path.join(save_directory,usage,filename),cutout_)
                            with open(os.path.join(save_directory,usage,filename_annotation), 'w') as bbox_text:
                                for b in bbox:
                                    bbox_text.write(",".join(map(str, b))+"\n")
                            if usage == "train":
                                with open(train_imglist_text,'a') as train_list:
                                    train_list.write('{0:010d}'.format(img_number)+"\n")
                            else:
                                with open(test_imglist_text,'a') as test_list:
                                    test_list.write('{0:010d}'.format(img_number)+"\n")


cutout_size = math.floor(output_size / scale) if scale != None else output_size

for directory in process_directories:
    print("current directory:"+directory)
    filelist = os.listdir(directory)
    image_list = []
    image_mask_list = []
    for file in filelist:
        root, ext = os.path.splitext(file)
        if ext == ".png" and root.find("Annotated") == -1:
            image_list.append(os.path.join(directory, file))
            image_mask_list.append(os.path.join(directory, root + "_Annotated_Cars.png"))

    for image_path, image_mask_path in zip(image_list,image_mask_list):
        print("processing image:" + image_path)
        make_img_cutouts(image_path, image_mask_path, save_directory, cutout_size, output_size,windowsize)

