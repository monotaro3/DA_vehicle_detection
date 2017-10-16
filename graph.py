# coding: utf-8

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def gengraph(logfile,savedir="graph",figname = "train_graph.png",mode="SSD"):
    with open(logfile) as f:
        # read json file
        ch_log = json.load(f)

    if mode =="SSD":
        loss = [ep['main/loss'] for ep in ch_log]
        loss_loc = [ep['main/loss/loc'] for ep in ch_log]
        loss_conf = [ep['main/loss/conf'] for ep in ch_log]
    elif mode =="ADDA":
        loss_t_enc = [ep['loss_t_enc'] for ep in ch_log]
        loss_dis = [ep['loss_dis'] for ep in ch_log]
    itr = [ep['iteration'] for ep in ch_log]

    if not os.path.isdir(savedir): os.makedirs(savedir)
    savepath = os.path.join(savedir,figname)

    plt.figure()
    plt.title("Training loss")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    if mode == "SSD":
        plt.plot(itr, loss, label="Loss")
        plt.plot(itr, loss_loc, label="Loss_loc")
        plt.plot(itr, loss_conf, label="Loss_conf")
    elif mode == "ADDA":
        plt.plot(itr, loss_t_enc, label="Loss_t_enc")
        plt.plot(itr, loss_dis, label="Loss_dis")

    plt.ylim([0,0.5])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.savefig(savepath)
    # plt.show()

if __name__ == "__main__":
    gengraph("log",figname="train_log_300_0.3_daug.png",mode="SSD")

