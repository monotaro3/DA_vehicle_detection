# coding: utf-8

import cv2 as cv
import numpy as np
import os

process_directories = []
process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Potsdam_ISPRS")
process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Selwyn_LINZ")
process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Toronto_ISPRS")
process_directories.append("E:/work/vehicle_detection_dataset/cowc/datasets/ground_truth_sets/Utah_AGRC")

save_directory = "E:/work/vehicle_detection_dataset/cowc_processed"

cutout_size = 512
windowsize = 50
adjust_margin = False
margin = 0
use_edge = False
rescale = 0 # 0: no rescaling 0.5 -> halve the resolution

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

def decode_mask2bbox(maskimg):
    global windowsize
    rp = np.where(maskimg == 255)
    h,w,c = maskimg.shape
    dots_ = []
    dot_num = rp[0].size
    for i in range(dot_num):
        dots_.append([rp[0][i]+1, rp[1][i]+1])
    bbox = []
    window_half = int(windowsize / 2)
    for d in dots_:
        xmin = d[1] - window_half
        ymin = d[0] - window_half
        xmax = d[1] + window_half
        ymax = d[0] + window_half
        if windowsize % 2 == 0:
            xmin +=1
            ymin +=1
            if xmin < 1: xmin = 1
            if ymin < 1: ymin = 1
            if xmax > w: xmax = w
            if ymax > h: ymax = h
        bbox.append([xmin,ymin,xmax,ymax])
    return bbox

def make_img_cutouts(image_path,image_mask_path,save_directory,cutout_size):
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
                        bbox = decode_mask2bbox(cutout_mask_)
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
        make_img_cutouts(image_path,image_mask_path,save_directory,cutout_size)

