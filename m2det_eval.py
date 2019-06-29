import os
import pickle
import numpy as np
import cv2 as cv
from chainercv import utils
from utils import make_bboxeslist_chainercv, draw_rect
from COWC_dataset_processed import vehicle_classes
from eval_detection_voc_custom import eval_detection_voc_custom

testimg_dir = "E:/work/dataset/experiments/vehicle_detection_dataset/NTT_test"
m2det_result = "E:/work/experiments/results/NTT_test_0.1_nonms"

nms_thresh = 0.45
normal_nms = True
low_conf_threshold_advance = 0.6 #set 0 to turn off
low_conf_threshold = 0. #set 0 to turn off

img_files = sorted((os.path.join(testimg_dir,fname) for fname in os.listdir(testimg_dir) if os.path.splitext(fname)[-1] in ('.jpg','.png','.tif')))
img_vis_files = sorted((os.path.join(m2det_result,os.path.splitext(fname)[0]+"_vis.jpg") for fname in os.listdir(testimg_dir) if os.path.splitext(fname)[-1] in ('.jpg','.png','.tif')))
gt_files = sorted((os.path.join(testimg_dir,os.path.splitext(fname)[0]+".txt") for fname in os.listdir(testimg_dir) if os.path.splitext(fname)[-1] in ('.jpg','.png','.tif')))
result_files = sorted((os.path.join(m2det_result, os.path.splitext(fname)[0]+"_det_result.pkl") for fname in os.listdir(testimg_dir) if os.path.splitext(fname)[-1] in ('.jpg','.png','.tif')))

bboxes = []
labels = []
scores = []
gt_bboxes = []
gt_labels = []

for i, r_file in enumerate(result_files):
    gt_bbox = make_bboxeslist_chainercv(gt_files[i])
    gt_bboxes.append(gt_bbox)
    gt_labels.append(np.stack([0] * len(gt_bbox)).astype(np.int32))
    if os.path.isfile(result_files[i]):
        with open(result_files[i], 'rb') as f:
            result = pickle.load(f)
        bbox = result[0].astype(np.float32)
        bbox[:, 0], bbox[:, 1] = bbox[:, 1], bbox[:, 0].copy()
        bbox[:, 2], bbox[:, 3] = bbox[:, 3], bbox[:, 2].copy()
        score = result[1].astype(np.float32)
        label = result[2].astype(np.int32)
        label[...] = 0  # car

        if low_conf_threshold_advance > 0:
            index = score > low_conf_threshold_advance
            bbox = bbox[index]
            label = label[index]
            score = score[index]

        if normal_nms:
            bbox_nms = list()
            label_nms = list()
            score_nms = list()

            # label-wise nms
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

        if low_conf_threshold > 0:
            index = score > low_conf_threshold
            bbox = bbox[index]
            label = label[index]
            score = score[index]
    else:
        bbox = np.empty((0,4)).astype(np.float32)
        score = np.empty((0,)).astype(np.float32)
        label = np.empty((0,)).astype(np.int32)

    bboxes.append(bbox)
    scores.append(score)
    labels.append(label)

result, stats, matches, selec_list = eval_detection_voc_custom(bboxes,labels,scores,gt_bboxes,gt_labels,iou_thresh=0.4)
mean_ap_f1 = (result['map'] + (stats[0]['F1'] if stats[0]['F1'] != None else 0)) / 2

for i, i_file in enumerate(img_files):
    img = cv.imread(i_file)
    draw_rect(img, bboxes[i], matches[i])
    gt_bbox = gt_bboxes[i]
    undetected_gt = gt_bbox[selec_list[i] == False]
    draw_rect(img, undetected_gt, np.array((0,) * undetected_gt.shape[0], dtype=np.int8), mode="GT")
    cv.imwrite(img_vis_files[i], img)

print("mean ap and f1:{}".format(mean_ap_f1))
print(result)
print(stats)

pass

