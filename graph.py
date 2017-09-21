# coding: utf-8

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def gengraph(logfile,savedir="graph",figname = "train_graph.png"):
    with open(logfile) as f:
        # read json file
        ch_log = json.load(f)
    loss = [ep['main/loss'] for ep in ch_log]
    loss_loc = [ep['main/loss/loc'] for ep in ch_log]
    loss_conf = [ep['main/loss/conf'] for ep in ch_log]
    itr = [ep['iteration'] for ep in ch_log]

    if not os.path.isdir(savedir): os.makedirs(savedir)
    savepath = os.path.join(savedir,figname)

    plt.figure()
    plt.title("Training loss")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.ylim([0,0.5])
    plt.plot(itr, loss, label="Loss")
    plt.plot(itr, loss_loc, label="Loss_loc")
    plt.plot(itr, loss_conf, label="Loss_conf")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.savefig(savepath)
    # plt.show()

if __name__ == "__main__":
    gengraph("log",figname="train_log_300_0.3.png")

