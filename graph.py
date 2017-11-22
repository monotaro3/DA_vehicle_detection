# coding: utf-8

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

def gengraph(logfile,savedir="graph",figname = "train_graph.png",mode="SSD",key_select = None):
    with open(logfile) as f:
        # read json file
        ch_log = json.load(f)

    if mode =="SSD":
        # loss = [ep['main/loss'] for ep in ch_log]
        # loss_loc = [ep['main/loss/loc'] for ep in ch_log]
        # loss_conf = [ep['main/loss/conf'] for ep in ch_log]
        keys = ['main/loss','main/loss/loc','main/loss/conf']
    elif mode =="ADDA":
        # loss_t_enc = [ep['loss_t_enc'] for ep in ch_log]
        # loss_dis = [ep['loss_dis'] for ep in ch_log]
        keys = ['loss_t_enc','loss_dis']
    elif mode == "DA_loss":
        keys = ['loss_cls', "loss_t_enc","loss_dis"]
    elif mode == "DA_eval":
        keys = ["validation/main/map", "validation/main/F1/car"]
    elif mode == "CORAL_loss":
        keys = ['main/cls_loss','main/CORAL_loss_weighted']
    elif mode == "CORAL_eval":
        keys = ['validation_1/main/map','validation_1/main/F1/car']
    data = defaultdict(lambda :[[],[]])
    for key in keys:
        for ep in ch_log:
            try:
                data[key][1].append(ep[key])
                data[key][0].append(ep['iteration'])
            except KeyError:
                pass
    # itr = [ep['iteration'] for ep in ch_log]

    if mode == "CORAL_eval":
        map_ = data['validation_1/main/map'][:]
        f1_ = data['validation_1/main/F1/car'][:]
        difference = set(map_[0]) ^ set(f1_[0])
        for d in difference:
            if d in map_[0]:
                index = map_[0].index(d)
                map_[0].pop(index)
                map_[1].pop(index)
            if d in f1_[0]:
                index = f1_[0].index(d)
                f1_[0].pop(index)
                f1_[1].pop(index)
        for i in range(len(map_[0])):
            data['mean_ap_F1'][0].append(map_[0][i])
            data['mean_ap_F1'][1].append((map_[1][i] + f1_[1][i]) / 2)

    if mode == "DA_eval":
        map_ = data['validation/main/map'][:]
        f1_ =  data['validation/main/F1/car'][:]
        difference = set(map_[0])^set(f1_[0])
        for d in difference:
            if d in map_[0]:
                index = map_[0].index(d)
                map_[0].pop(index)
                map_[1].pop(index)
            if d in f1_[0]:
                index = f1_[0].index(d)
                f1_[0].pop(index)
                f1_[1].pop(index)
        for i in range(len(map_[0])):
            data['mean_ap_F1'][0].append(map_[0][i])
            data['mean_ap_F1'][1].append((map_[1][i]+f1_[1][i])/2)

    if not os.path.isdir(savedir): os.makedirs(savedir)
    savepath = os.path.join(savedir,figname)

    plt.figure(figsize=(10,6))
    #plt.figure()
    plt.title("Training loss")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    # if mode == "SSD":
    #     plt.plot(itr, loss, label="Loss")
    #     plt.plot(itr, loss_loc, label="Loss_loc")
    #     plt.plot(itr, loss_conf, label="Loss_conf")
    # elif mode == "ADDA":
    #     plt.plot(itr, loss_t_enc, label="Loss_t_enc")
    #     plt.plot(itr, loss_dis, label="Loss_dis")
    # map = np.array(data["validation/main/map"])
    # f1 = np.array(data["validation/main/F1/car"])
    #index_sorted_
    # try:
    #     map = data["validation/main/map"]
    #     f1 = data["validation/main/F1/car"]
    #     mean = []
    #     for i in range(len(map)):
    #         if map[i] != None and f1[i] != None:
    #             mean.append((i,(map[i]+f1[i])/2))
    #     mean.sort(key=lambda x: x[1],reverse=True)
    #     for j in range(10):
    #         print("map:{0},f1:{1},mean:{2} iter{3}\n".format(map[mean[j][0]],f1[mean[j][0]],mean[j][1],itr[mean[j][0]]))
    # except KeyError:
    #     pass
    # for key in data.keys():
    #     d_array = np.array(data[key])
    #     index_sorted = np.argsort(d_array)[::-1]
    #     print("top arguments:")
    #     print(index_sorted[0:10])
    #     print("top values:")
    #     print(d_array[index_sorted[0:10]])

    if key_select:
        for key in key_select:
            plt.plot(data[key][0], data[key][1],
                     label=key.replace("main/", "").replace("validation/", "").replace("validation_1/", ""))
    else:
        for key in data.keys():
            plt.plot(data[key][0], data[key][1], label=key.replace("main/","").replace("validation/","").replace("validation_1/",""))

    #plt.ylim([0.,10])
    plt.ylim([0.5, 0.9])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.6)
    plt.savefig(savepath)
    # plt.show()

if __name__ == "__main__":
    gengraph("model/DA/NTT_buf_sfixed_alt_100_nalign_DA2_dispt_20000/log",savedir='model/DA/NTT_buf_sfixed_alt_100_nalign_DA2_dispt_20000/',figname="train_eval.png",mode="DA_eval") #,key_select=('mean_ap_F1',))
