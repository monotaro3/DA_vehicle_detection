# coding: utf-8

import numpy as np
from osgeo import gdal, osr
from chainer import serializers
import os
from chainercv import utils as utils_chainercv
import chainer
from chainercv.links.model.ssd import VGG16Extractor300
import pickle
import cv2 as cv
import math

def gen_dms_time_str(time_sec):
    sec = time_sec % 60
    min = (int)(time_sec / 60) % 60
    hour = (int)(time_sec / 60 / 60) % 24
    day = (int)(time_sec / 60 / 60 / 24)
    time_str = ""
    if day!=0: time_str = str(day) + " day(s)"
    if time_str != "" or hour != 0: time_str = time_str + " " + str(hour) + " hour(s)"
    if time_str != "" or min != 0: time_str = time_str + " " + str(min) + " min(s)"
    time_str = time_str + (" " if time_str != "" else "") + str(sec) + " sec(s)"
    return time_str

def make_bboxeslist_chainercv(gt_file):
    gt_txt = open(gt_file,"r")
    bboxes = []
    line = gt_txt.readline()
    while (line):
        xmin, ymin, xmax, ymax = line.split(",")
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        bboxes.append([ymin-1, xmin-1, ymax-1, xmax-1])
        line = gt_txt.readline()
    bboxes = np.stack(bboxes).astype(np.float32)
    return bboxes

def convertDepth(input,output):
    gdal.AllRegister()
    src = gdal.Open(input)
    SpaRef = src.GetProjection()
    geoTransform = src.GetGeoTransform()

    cols = src.RasterXSize
    rows = src.RasterYSize
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output, cols, rows, 3, gdal.GDT_Byte)
    dst.SetGeoTransform(geoTransform)
    dstSRS = osr.SpatialReference()
    dstSRS.ImportFromWkt(SpaRef)
    dst.SetProjection(dstSRS.ExportToWkt())
    for i in range(1,4):
        src_data_ = src.GetRasterBand(i).ReadAsArray()
        src_data = src_data_.copy()
        src_data[src_data >= 1024] = 1023
        src_data = (src_data/4).astype(np.uint8)
        dst.GetRasterBand(i).WriteArray(src_data)
    dst = None

def decode_mask2bbox(maskimg,windowsize):
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

def convertImgs2fmaps(imgsdir,savedir,extractor_path,dirname="train"):
    num_example = 0
    files = os.listdir(imgsdir)
    imgs = []
    for f in files:
        root, ext = os.path.splitext(f)
        if ext in [".png", ".tif"]:
            imgs.append(os.path.join(imgsdir,f))
    extractor = VGG16Extractor300()
    serializers.load_npz(extractor_path,extractor)
    extractor.to_gpu()
    fmaplist = os.path.join(savedir,"list",dirname+".txt")
    if not os.path.isdir(os.path.join(savedir,dirname)): os.makedirs(os.path.join(savedir,dirname))
    if not os.path.isdir(os.path.join(savedir, "list")):os.makedirs(os.path.join(savedir, "list"))
    for imgpath in imgs:
        num_example += 1
        if num_example % 100 == 1: print("processing {0}th file...".format(num_example))
        img = utils_chainercv.read_image(imgpath, color=True)
        img = img[np.newaxis, :]
        img = chainer.cuda.to_gpu(img)
        with chainer.no_backprop_mode():
            fmap_ = extractor(img)
        fmap = []
        for i in range(6):
            fmap.append(chainer.cuda.to_cpu(fmap_[i][0].data))
        dir, file = os.path.split(imgpath)
        root, ext = os.path.splitext(file)
        with open(fmaplist, 'a') as listfile:
            listfile.write(root + "\n")
        with open(os.path.join(savedir,dirname,root+".fmp"),"wb") as f:
            pickle.dump(fmap,f)

def make_img_cutout(imgdir,savedir,imgsize,margin = 0,useEdge=False):
    num_example = 0
    savedir_img = "train"
    savedir_list = "list"
    if not os.path.isdir(os.path.join(savedir,savedir_img)):os.makedirs(os.path.join(savedir,savedir_img))
    if not os.path.isdir(os.path.join(savedir, savedir_list)): os.makedirs(os.path.join(savedir, savedir_list))
    listfile_path = os.path.join(savedir,savedir_list,savedir_img+".txt")
    files = os.listdir(imgdir)
    imgs = []
    for f in files:
        root, ext = os.path.splitext(f)
        if ext in [".tif",".png","jpg"]:
            imgs.append(os.path.join(imgdir,f))
    for imgpath in imgs:
        img = cv.imread(imgpath)
        H, W, c = img.shape
        W_slot = int(math.ceil((W-margin)/(imgsize-margin)))
        H_slot = int(math.ceil((H-margin)/(imgsize-margin)))
        root, ext = os.path.splitext(imgpath)
        for h in range(H_slot):
            if h == H_slot-1:
                if H == (h+1) * (imgsize-margin) + margin or useEdge:
                    h_start, h_end = -imgsize, None
                else: continue
            else:
                h_start, h_end = h * (imgsize-margin), h * (imgsize-margin) + imgsize
            for w in range(W_slot):
                if w == W_slot - 1:
                    if W == (w + 1) * (imgsize - margin) + margin or useEdge:
                        w_start, w_end = -imgsize, None
                    else:
                        continue
                else:
                    w_start, w_end = w * (imgsize - margin), w * (imgsize - margin) + imgsize
                cutout = img[h_start:h_end,w_start:w_end,:]
                num_example += 1
                if num_example % 100 == 0: print("{0}th file has been output.".format(num_example))
                rootname = "{0:010d}".format(num_example)
                cv.imwrite(os.path.join(savedir,savedir_img,rootname+ext),cutout)
                with open(listfile_path, 'a') as list:
                    list.write(rootname + "\n")


if __name__ == "__main__":
    #convertDepth('c:/work/spacenet\\raw\\RGB-PanSharpen_AOI_5_Khartoum_img1.tif',"c:/work/test4.tif")
    # trainer = None
    # serializers.load_npz("model/snapshot_iter_30000", trainer)
    # pass
    #convertImgs2fmaps("E:/work/vehicle_detection_dataset/cowc_300px_0.3/train","E:/work/vehicle_detection_dataset/cowc_300px_0.3_fmap","model/vgg_300_0.3_30000")
    make_img_cutout("E:/work/vehicle_detection_dataset/Khartoum_adda_raw","E:/work/vehicle_detection_dataset/Khartoum_adda",300)