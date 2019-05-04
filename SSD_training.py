# coding: utf-8

import argparse
import copy
import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainer.dataset import convert

# from chainercv.datasets import voc_detection_label_names
# from chainercv.datasets import VOCDetectionDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

import time

#--custom
from SSD_for_vehicle_detection import SSD300_vd, SSD512_vd, defaultbox_size_300, defaultbox_size_512
from COWC_dataset_processed import COWC_dataset_processed, vehicle_classes
from utils import gen_dms_time_str







class ConcatenatedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, *datasets):
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def get_example(self, i):
        if i < 0:
            raise IndexError
        for dataset in self._datasets:
            if i < len(dataset):
                return dataset[i]
            i -= len(dataset)
        raise IndexError


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # # 2. Random expansion
        # if np.random.randint(2):
        #     img, param = transforms.random_expand(
        #         img, fill=self.mean, return_param=True)
        #     bbox = transforms.translate_bbox(
        #         bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])
        #
        # # 3. Random cropping
        # img, param = random_crop_with_bbox_constraints(
        #     img, bbox, return_param=True)
        # bbox, param = transforms.crop_bbox(
        #     bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
        #     allow_outside_center=False, return_param=True)
        # label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # # 5. Random horizontal flipping
        # img, params = transforms.random_flip(
        #     img, x_random=True, return_param=True)
        # bbox = transforms.flip_bbox(
        #     bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label

class updater_st(chainer.training.StandardUpdater):
    def __init__(self, iterator,optimizer,gpu):
        super(updater_st, self).__init__(iterator, optimizer,device=gpu)

    def update_core(self):
        batch = self._iterators['main'].next()
        batch.extend(self._iterators['target'].next())
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            in_vars = tuple(chainer.variable.Variable(x) for x in in_arrays)
            optimizer.update(loss_func, *in_vars)
        else:
            in_var = chainer.variable.Variable(in_arrays)
            optimizer.update(loss_func, in_var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument(
        '--resolution', type=float, choices=(0.15,0.16,0.3), default=0.15)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iteration', type=int, default=120000)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--out_progress')
    parser.add_argument('--resume')
    parser.add_argument('--resumemodel')
    parser.add_argument('--datadir')
    parser.add_argument('--target_data')
    parser.add_argument('--s_t_ratio',type=int, nargs=2)
    parser.add_argument('--weightdecay', type=float,  default=0.0005)
    parser.add_argument('--lrdecay_schedule', nargs = '*', type=int, default=[80000,100000])
    parser.add_argument('--snapshot_interval', type=int)
    args = parser.parse_args()

    batchsize = args.batchsize
    gpu = args.gpu
    out = args.out
    resume = args.resume

    exectime = time.time()

    # if args.model == 'ssd300':
    #     model = SSD300(
    #         n_fg_class=len(voc_detection_label_names),
    #         pretrained_model='imagenet')
    # elif args.model == 'ssd512':

    # model = SSD300(
    #     n_fg_class=len(vehicle_classes),
    #     pretrained_model='imagenet')

    if args.model == 'ssd300':
        model = SSD300_vd(
            n_fg_class=len(vehicle_classes),
            pretrained_model='imagenet', defaultbox_size=defaultbox_size_300[args.resolution])
    else:
        model = SSD512_vd(
            n_fg_class=len(vehicle_classes),
            pretrained_model='imagenet',defaultbox_size=defaultbox_size_512[args.resolution])


    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    if args.target_data:
        if args.s_t_ratio:
            if args.s_t_ratio[0] == 0:
                s_train = TransformDataset(
                    COWC_dataset_processed(split="train", datadir=args.target_data),
                    Transform(model.coder, model.insize, model.mean))
                s_train_iter = chainer.iterators.MultiprocessIterator(s_train, batchsize)
            else:
                s_batch = int(args.batchsize * args.s_t_ratio[0]/sum(args.s_t_ratio))
                t_batch = args.batchsize - s_batch
                if s_batch <= 0:
                    print("invalid batchsize")
                    exit(0)
                s_train = TransformDataset(
                    COWC_dataset_processed(split="train", datadir=args.datadir),
                    Transform(model.coder, model.insize, model.mean))
                t_train = TransformDataset(
                    COWC_dataset_processed(split="train", datadir=args.target_data),
                    Transform(model.coder, model.insize, model.mean))
                s_train_iter = chainer.iterators.MultiprocessIterator(s_train, s_batch)
                t_train_iter = chainer.iterators.MultiprocessIterator(t_train, t_batch)
        else:
            s_train = TransformDataset(
                ConcatenatedDataset(
                    COWC_dataset_processed(split="train", datadir=args.datadir),
                    COWC_dataset_processed(split="train", datadir=args.target_data)
                ),
                #COWC_dataset_processed(split="train", datadir=args.datadir),
                Transform(model.coder, model.insize, model.mean))
            s_train_iter = chainer.iterators.MultiprocessIterator(s_train, batchsize)
    else:
        s_train = TransformDataset(
            # ConcatenatedDataset(
            #     VOCDetectionDataset(year='2007', split='trainval'),
            #     VOCDetectionDataset(year='2012', split='trainval')
            # ),
            COWC_dataset_processed(split="train",datadir=args.datadir),
            Transform(model.coder, model.insize, model.mean))
        s_train_iter = chainer.iterators.MultiprocessIterator(s_train, batchsize)

    # test = VOCDetectionDataset(
    #     year='2007', split='test',
    #     use_difficult=True, return_difficult=True)
    test = COWC_dataset_processed(split="validation",datadir=args.datadir)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)

    # initial lr is set to 1e-3 by ExponentialShift
    if args.lrdecay_schedule[0] == -1:
        optimizer = chainer.optimizers.MomentumSGD(args.lr)
    else:
        optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(args.weightdecay))

    if args.target_data and args.s_t_ratio and args.s_t_ratio[0] != 0:
        updater = updater_st({'main': s_train_iter, 'target':t_train_iter}, optimizer, gpu)
    else:
        updater = training.StandardUpdater(s_train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out)

    if args.lrdecay_schedule[0] != -1:
        trainer.extend(
            extensions.ExponentialShift('lr', 0.1, init=args.lr),
            trigger=triggers.ManualScheduleTrigger(list(args.lrdecay_schedule), 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=vehicle_classes),
        trigger=(10000, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']),
        trigger=log_interval)

    progress_args = {"update_interval":10}
    if args.out_progress:
        fo = open(args.out_progress, 'w')
        progress_args["out"] = fo

    trainer.extend(extensions.ProgressBar(**progress_args))

    trainer.extend(extensions.snapshot(), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(args.iteration, 'iteration'))

    if resume:
        serializers.load_npz(resume, trainer)

    # serializers.load_npz("model/snapshot_iter_300_0.16_55000", trainer)
    # serializers.save_npz("model/ssd_300_0.16_55000",trainer.updater._optimizers["main"].target.model)

    if args.resumemodel:
        serializers.load_npz(args.resumemodel, model)

    trainer.run()

    if args.out_progress:
        fo.close()

    exectime = time.time() - exectime
    exectime_str = gen_dms_time_str(exectime)
    print("exextime:"+exectime_str)

if __name__ == '__main__':
    main()