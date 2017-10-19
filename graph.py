# coding: utf-8

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

def gengraph(logfile,savedir="graph",figname = "train_graph.png",mode="SSD"):
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
    data = defaultdict(list)
    for key in keys:
        for ep in ch_log:
            try:
                data[key].append(ep[key])
            except KeyError:
                data[key].append(0)
    itr = [ep['iteration'] for ep in ch_log]

    if not os.path.isdir(savedir): os.makedirs(savedir)
    savepath = os.path.join(savedir,figname)

    plt.figure(figsize=(30,10))
    plt.figure()
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
    for key in data.keys():
        d_array = np.array(data[key])
        index_sorted = np.argsort(d_array)[::-1]
        print("top arguments:")
        print(index_sorted[0:10])
        print("top values:")
        print(d_array[index_sorted[0:10]])

    for key in data.keys():
        plt.plot(itr, data[key], label=key)

    plt.ylim([0,1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.savefig(savepath)
    # plt.show()

if __name__ == "__main__":
    gengraph("log",figname="train_log_300_0.3_daug_DA_map_NTT.png",mode="DA_eval")

