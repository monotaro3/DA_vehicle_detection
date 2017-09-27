# coding: utf-8

import numpy as np
from osgeo import gdal, osr
from chainer import serializers

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

if __name__ == "__main__":
    convertDepth('c:/work/spacenet\\raw\\RGB-PanSharpen_AOI_5_Khartoum_img1.tif',"c:/work/test4.tif")
    # trainer = None
    # serializers.load_npz("model/snapshot_iter_30000", trainer)
    # pass