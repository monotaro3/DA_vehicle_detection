# coding: utf-8

import matplotlib.pyplot as plot
from chainer import serializers
from chainercv import utils
import math
import numpy as np
from chainercv.visualizations import vis_bbox
from eval_detection_voc_custom import eval_detection_voc_custom
import cv2 as cv

#--custom
from SSD_for_vehicle_detection import SSD300_vd, SSD512_vd
from COWC_dataset_processed import vehicle_classes
from SSD_training import defaultbox_size_300, defaultbox_size_512
from utils import make_bboxeslist_chainercv

def draw_rect(image, bbox, match):
    for i in range(bbox.shape[0]):
        color = (0, 0, 255) if match[i] == 1 else (255, 0, 0)
        cv.rectangle(image, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), color)

def ssd_predict(model, image, margin,nms_thresh = 0.45):
    size = model.insize
    c, H, W = image.shape

    stride = size-margin
    H_slot = math.ceil((H - margin) / stride)
    W_slot = math.ceil((W - margin) / stride)

    bbox = list()
    label = list()
    score = list()

    for h in range(H_slot):
        offset_H = stride * h if h < H_slot-1 else H - size
        for w in range(W_slot):
            offset_W = stride * w if w < W_slot-1 else W - size
            cutout = image[:,offset_H:offset_H+size,offset_W:offset_W+size]
            bboxes, labels, scores = model.predict([cutout])
            bbox_, label_, score_ = bboxes[0], labels[0], scores[0]
            bbox_[:,(0,2)] += offset_H # bbox_: (y_min, x_min, y_max, x_max)
            bbox_[:, (1, 3)] += offset_W
            bbox.append(bbox_)
            label.append(label_)
            score.append(score_)
    bbox = np.vstack(bbox).astype(np.float32)
    label = np.hstack(label).astype(np.int32)
    score = np.hstack(score).astype(np.float32)

    bbox_nms = list()
    label_nms = list()
    score_nms = list()

    #label-wise nms
    for l in range(len(vehicle_classes)):
        mask_l = label == l
        bbox_l = bbox[mask_l]
        score_l = score[mask_l]
        indices = utils.non_maximum_suppression(
            bbox_l, nms_thresh, score_l)
        bbox_l = bbox_l[indices]
        score_l = score_l[indices]
        bbox_nms.append(bbox_l)
        label_nms.append(np.array((l,) * len(bbox_l)))
        score_nms.append(score_l)
    bbox = np.vstack(bbox_nms).astype(np.float32)
    label = np.hstack(label_nms).astype(np.int32)
    score = np.hstack(score_nms).astype(np.float32)

    return bbox, label, score

def main(ssd_path,imagepath,modelsize="ssd300",resolution=0.16):
    margin = 50
    gpu = 0

    if modelsize == 'ssd300':
        model = SSD300_vd(
            n_fg_class=len(vehicle_classes),
            defaultbox_size=defaultbox_size_300[resolution])
    else:
        model = SSD512_vd(
            n_fg_class=len(vehicle_classes),
            defaultbox_size=defaultbox_size_512[resolution])

    serializers.load_npz(ssd_path, model)
    image = utils.read_image(imagepath, color=True)

    if gpu >= 0: model.to_gpu()

    bbox, label, score = ssd_predict(model,image,margin)

    gt_bbox = make_bboxeslist_chainercv("E:/work/vehicle_detection_dataset/cowc_processed/train/0000000001.txt")
    gt_label = np.stack([0]*len(gt_bbox)).astype(np.int32)

    result, stats, matches = eval_detection_voc_custom([bbox],[label],[score],[gt_bbox],[gt_label])

    #visualizations
    vis_bbox(
        image, bbox, label, score, label_names=vehicle_classes)
    #plot.show()
    plot.savefig("result/vis1.png")

    image_ = image.copy()
    image_ = image_.transpose(1,2,0)
    image_ = cv.cvtColor(image_, cv.COLOR_RGB2BGR)
    draw_rect(image_,bbox,matches[0])
    cv.imwrite("result/vis2.png",image_)

if __name__ == "__main__":
    imagepath = "E:/work/vehicle_detection_dataset/cowc_processed/train/0000000001.png"
    modelpath = "model/ssd_300_0.16_120000"
    main(modelpath,imagepath)


