# coding: utf-8

import matplotlib.pyplot as plot
import chainer
from chainer import serializers
from chainer import reporter
from chainercv import utils
import math
import os
import numpy as np
from chainercv.visualizations import vis_bbox
from eval_detection_voc_custom import eval_detection_voc_custom
import cv2 as cv
import csv

#--custom
from SSD_for_vehicle_detection import SSD300_vd, SSD512_vd
from COWC_dataset_processed import vehicle_classes
from SSD_for_vehicle_detection import defaultbox_size_300, defaultbox_size_512
from utils import make_bboxeslist_chainercv

def draw_rect(image, bbox, match,mode="TEST"):
    for i in range(bbox.shape[0]):
        if mode == "TEST":
            color = (0, 0, 255) if match[i] == 1 else (255, 0, 0)
        if mode == "GT":
            color = (0, 255, 0)
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

def mask_trim(maskimg,bbox, label,score):
    bbox_center_y = ((bbox[:,0] + bbox[:,2])/2).astype(np.int)
    bbox_center_x = ((bbox[:,1] + bbox[:,3])/2).astype(np.int)
    center_values = maskimg[bbox_center_y,bbox_center_x,2]
    inner_mask = center_values == 255
    return bbox[inner_mask], label[inner_mask], score[inner_mask]

def ssd_test(ssd_model, imagepath, modelsize="ssd300", resolution=0.16, procDir=False, testonly = False, resultdir ="result", evalonly=False, mask=False):
    margin = 50
    # gpu = 0

    images = []
    gt_files = []

    if (not evalonly) and (not os.path.isdir(resultdir)):
        os.makedirs(resultdir)

    if procDir:
        if not os.path.isdir(imagepath):
            try:
                raise(ValueError("invalid image directory path"))
            except ValueError as e:
                print(e)
                return
        files = os.listdir(imagepath)
        for f in files:
            root, ext = os.path.splitext(f)
            if ext == ".tif":
                images.append(os.path.join(imagepath,f))
                if not testonly:
                    gt_files.append(os.path.join(imagepath,root+".txt"))
    else:
        images.append(imagepath)
        if not testonly:
            root, ext = os.path.splitext(imagepath)
            gt_files.append(root + ".txt")

    if not isinstance(ssd_model,chainer.link.Link):
        if modelsize == 'ssd300':
            model = SSD300_vd(
                n_fg_class=len(vehicle_classes),
                defaultbox_size=defaultbox_size_300[resolution])
        else:
            model = SSD512_vd(
                n_fg_class=len(vehicle_classes),
                defaultbox_size=defaultbox_size_512[resolution])

        serializers.load_npz(ssd_model, model)
    else:
        model = ssd_model
    # if gpu >= 0: model.to_gpu()
    if model.xp == np: model.to_gpu()

    #predict
    bboxes = []
    labels = []
    scores = []
    gt_bboxes = []
    gt_labels = []
    result_stat = []
    for i in range(len(images)):
        image = utils.read_image(images[i], color=True)
        bbox, label, score = ssd_predict(model,image,margin)
        if mask:
            root, ext = os.path.splitext(images[i])
            mask_file = root + "_mask.png"
            if os.path.isfile(mask_file):
                maskimg = cv.imread(mask_file)
                bbox, label, score = mask_trim(maskimg,bbox,label,score)
            else:
                print("No mask file:" + mask_file)
        bboxes.append(bbox)
        labels.append(label)
        scores.append(score)
        if testonly:
            dirpath, fname = os.path.split(images[i])
            result_stat.append([fname, "number of detected cars",len(bbox)])
            result_stat.append([])
        else:
            gt_bbox = make_bboxeslist_chainercv(gt_files[i])
            gt_bboxes.append(gt_bbox)
            # labels are without background, i.e. class_labels.index(class). So in this case 0 means cars
            gt_labels.append(np.stack([0]*len(gt_bbox)).astype(np.int32))
            result_i, stats_i, matches_i, selec_i = eval_detection_voc_custom([bbox],[label],[score],[gt_bbox],[np.stack([0]*len(gt_bbox)).astype(np.int32)],iou_thresh=0.4)
            dirpath,fname = os.path.split(images[i])
            root, ext = os.path.splitext(os.path.join(resultdir,fname))
            result_stat.append([fname,"map",result_i["map"]])
            result_stat.append(['', "ap", float(result_i["ap"])])
            result_stat.append(['', "PR", stats_i[0]["PR"]])
            result_stat.append(['', "RR", stats_i[0]["RR"]])
            result_stat.append(['', "FAR", stats_i[0]["FAR"]])
            result_stat.append(['', "F1", stats_i[0]["F1"]])
            result_stat.append(['', "mean_ap_F1",  (result_i['map'] + (stats_i[0]['F1'] if stats_i[0]['F1'] != None else 0)) / 2])
            result_stat.append(['', "number of detected cars",len(bbox)])
            result_stat.append(['', "number of TP", np.sum(matches_i[0]==1)])
            result_stat.append(['', "number of FP", np.sum(matches_i[0] == 0)])
            result_stat.append(['', "number of gt", len(gt_bbox)])
            result_stat.append(['', "number of FN(undected gt)", np.sum(selec_i[0]==False)])
            # result_stat.append(['', "number of FN(undected gt)(calculated)", len(gt_bbox)-np.sum(matches_i[0]==1)])
            result_stat.append([])
            # resulttxt_i = root + "_result.txt"
            # with open(resulttxt_i, mode="w") as f:
            #     f.write(str(result_i) + "\n" + str(stats_i))

    if not testonly:
        result, stats, matches, selec_list = eval_detection_voc_custom(bboxes,labels,scores,gt_bboxes,gt_labels,iou_thresh=0.4)
        mean_ap_f1 = (result['map'] + (stats[0]['F1'] if stats[0]['F1'] != None else 0)) / 2

    if not evalonly:
        #visualizations
        for imagepath , bbox, label, score in zip(images,bboxes,labels,scores):
            dir, imagename = os.path.split(imagepath)
            result_name, ext = os.path.splitext(imagename)
            image = utils.read_image(imagepath, color=True)
            if mask:
                mask_inverted = np.ones(maskimg.shape,dtype=bool)
                mask_indices = np.where(maskimg==255)
                mask_inverted[mask_indices[0],mask_indices[1],:] = False
                mask_inverted = mask_inverted.transpose((2,0,1))
                image[mask_inverted] = (image[mask_inverted]/2).astype(np.int32)
            vis_bbox(
                image, bbox, label, score, label_names=vehicle_classes)
            #plot.show()
            plot.savefig(os.path.join(resultdir,result_name+ "_vis1.png"))

            #result
            image_ = image.copy()
            image_ = image_.transpose(1,2,0)
            image_ = cv.cvtColor(image_, cv.COLOR_RGB2BGR)
            if testonly:
                draw_rect(image_, bbox, np.array((1,) * bbox.shape[0], dtype=np.int8))
            else:
                draw_rect(image_,bbox,matches[images.index(imagepath)])
            cv.imwrite(os.path.join(resultdir,result_name+ "_vis2.png"),image_)
            gt_bbox = gt_bboxes[images.index(imagepath)]
            undetected_gt = gt_bbox[selec_list[images.index(imagepath)]==False]
            draw_rect(image_, undetected_gt, np.array((0,) * undetected_gt.shape[0], dtype=np.int8), mode="GT")
            cv.imwrite(os.path.join(resultdir, result_name + "_vis3.png"), image_)

            #gt visualization
            if not testonly:
                image_ = image.copy()
                image_ = image_.transpose(1, 2, 0)
                image_ = cv.cvtColor(image_, cv.COLOR_RGB2BGR)
                gt_bbox = gt_bboxes[images.index(imagepath)]
                draw_rect(image_, gt_bbox, np.array((0,) * gt_bbox.shape[0],dtype=np.int8),mode="GT")
                cv.imwrite(os.path.join(resultdir,result_name+ "_vis_gt.png"), image_)
    if not testonly:
        result_txt = os.path.join(resultdir, "result.txt")
        with open(result_txt,mode="w") as f:
            f.write(str(result) +"\n" + str(stats) + '\nmean_ap_F1: ' + str(mean_ap_f1))
        result_stat.append(["summary", "map", result["map"]])
        result_stat.append(['', "ap", float(result["ap"])])
        result_stat.append(['', "PR", stats[0]["PR"]])
        result_stat.append(['', "RR", stats[0]["RR"]])
        result_stat.append(['', "FAR", stats[0]["FAR"]])
        result_stat.append(['', "F1", stats[0]["F1"]])
        result_stat.append(
            ['', "mean_ap_F1", mean_ap_f1])
    with open(os.path.join(resultdir,"result_stat.csv"),"w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(result_stat)

    if not testonly:
        print(result)
        print(stats)
        print("mean_ap_F1:{0}".format(mean_ap_f1))

        return result, stats

class ssd_evaluator(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, img_dir, target,updater, savedir, n_ranking=5, resolution=0.3,modelsize="ssd300",evalonly=True, label_names=None,save_bottom = 0.6):
        super(ssd_evaluator, self).__init__(
            None, target)
        self.img_dir = img_dir
        self.resolution = resolution
        self.modelsize = modelsize
        self.label_names = label_names
        self.evalonly = evalonly
        self.save_bottom = save_bottom
        self.n_ranking = n_ranking
        self.rank_map =  []
        self.rank_map_data = np.full((self.n_ranking,3),-1.,dtype=np.float32)
        self.rank_F1 = []
        self.rank_F1_data = np.full((self.n_ranking, 3), -1., dtype=np.float32)
        self.rank_mean = []
        self.rank_mean_data = np.full((self.n_ranking, 4), -1., dtype=np.float32)
        self.updater = updater
        self.savedir = savedir

    def evaluate(self):
        target = self._targets['main']
        result, stats = ssd_test(target, self.img_dir, procDir=True, resolution=self.resolution,
                 modelsize=self.modelsize,evalonly=self.evalonly,resultdir=self.savedir)

        report = {'map': result['map']}
        for i in range(len(stats)):
            classname = self.label_names[i] if self.label_names is not None else "class"+str(i)
            report['PR/{:s}'.format(classname)] = stats[i]['PR']
            report['RR/{:s}'.format(classname)] = stats[i]['RR']
            report['FAR/{:s}'.format(classname)] = stats[i]['FAR']
            report['F1/{:s}'.format(classname)] = stats[i]['F1']

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = result['ap'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        current_iteration = self.updater.iteration
        save_flag = False
        del_iter = []
        mean_F1 = 0
        for i in range(len(stats)):
            if stats[i]['F1'] != None:
                mean_F1 += stats[i]['F1']
        mean_F1 /= len(stats)
        mean_map_mF1 = (result['map'] + mean_F1) / 2
        if len(self.rank_map) == 0:
            if result['map'] > self.save_bottom:
                save_flag = True
                self.rank_map.append([current_iteration, result['map'],mean_F1])
        elif result['map'] > self.rank_map[-1][1]:
            save_flag = True
            if len(self.rank_map) ==self.n_ranking:
                iter = self.rank_map.pop()[0]
                if not iter in del_iter: del_iter.append(iter)
            self.rank_map.append([current_iteration, result['map'],mean_F1])
            self.rank_map.sort(key=lambda x: x[1],reverse=True)
        if len(self.rank_F1) == 0:
            if mean_F1 > self.save_bottom:
                save_flag = True
                self.rank_F1.append([current_iteration, result['map'],mean_F1])
        elif mean_F1 > self.rank_F1[-1][2]:
            save_flag = True
            if len(self.rank_F1) == self.n_ranking:
                iter = self.rank_F1.pop()[0]
                if not iter in del_iter: del_iter.append(iter)
            self.rank_F1.append([current_iteration, result['map'],mean_F1])
            self.rank_F1.sort(key=lambda x: x[2],reverse=True)
        if len(self.rank_mean) == 0:
            if mean_map_mF1 > self.save_bottom:
                save_flag = True
                self.rank_mean.append([current_iteration, result['map'],mean_F1,mean_map_mF1])
        elif mean_map_mF1 > self.rank_mean[-1][3]:
            save_flag = True
            if len(self.rank_mean) ==self.n_ranking:
                iter = self.rank_mean.pop()[0]
                if not iter in del_iter: del_iter.append(iter)
            self.rank_mean.append([current_iteration, result['map'],mean_F1,mean_map_mF1])
            self.rank_mean.sort(key=lambda x: x[3],reverse=True)
        if save_flag:
            serializers.save_npz(os.path.join(self.savedir,target.__class__.__name__ + "_{0}.npz".format(current_iteration)),target)
        for iter in del_iter:
            if not iter in [i[0] for i in self.rank_map + self.rank_F1 + self.rank_mean]:
                os.remove(os.path.join(self.savedir,target.__class__.__name__ + "_{0}.npz".format(iter)))

        self.encode(self.rank_map,self.rank_map_data)
        self.encode(self.rank_F1, self.rank_F1_data)
        self.encode(self.rank_mean, self.rank_mean_data)

        ranking_summary = [["best map"],["iter","map","F1"]]
        ranking_summary.extend(self.rank_map)
        ranking_summary.extend([[],["best F1"], ["iter", "map", "F1"]])
        ranking_summary.extend(self.rank_F1)
        ranking_summary.extend([[], ["best mean"], ["iter", "map", "F1","mean"]])
        ranking_summary.extend(self.rank_mean)
        with open(os.path.join(self.savedir,"ranking.csv"),"w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(ranking_summary)

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation

    def encode(self,ranking_list,data):
        n_rank = len(ranking_list)
        if n_rank > 0:
            data[:n_rank] = np.array(ranking_list)

    def decode(self, data):
        ranking_list = []
        for i in range(len(data)):
            if data[i][0] == -1:
                break
            else:
                ranking_list.append([data[i][j] for j in range(data.shape[1])])
        return ranking_list

    def serialize(self, serializer):
        self.rank_map_data = serializer('rank_map', self.rank_map_data)
        self.rank_F1_data = serializer('rank_F1', self.rank_F1_data)
        self.rank_mean_data = serializer('rank_mean', self.rank_mean_data)
        if isinstance(serializer, chainer.serializer.Deserializer):
            self.rank_map = self.decode(self.rank_map_data)
            self.rank_F1 = self.decode(self.rank_F1_data)
            self.rank_mean = self.decode(self.rank_mean_data)

if __name__ == "__main__":
    imagepath = "c:/work/DA_images/NTT_scale0.3/2_6" #c:/work/DA_images/kashiwa_lalaport/0.3"#"#"E:/work/vehicle_detection_dataset/cowc_processed/train/0000000001.png"
    modelpath = "model/DA/NTT_buf_alt_100_nalign_DA4_nmargin/SSD300_vd_7000.npz" #"model/DA/CORAL/ft_patch_w100000000_nmargin/SSD300_vd_33750.npz"
    # modelpath = "../chainer-cyclegan/experiment_data/models/NTT_fake_GT_L1_l10/model_iter_50000"
    # result_dir = "../chainer-cyclegan/experiment_data/results/NTT_fake_GT_L1_l10/"
    result_dir = 'result/res0.3/re/rerun/adv_test'
    #test 
    ssd_test(modelpath,imagepath,procDir=True,resultdir=result_dir,resolution=0.3,modelsize="ssd300")


