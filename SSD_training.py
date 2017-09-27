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

from chainercv.datasets import voc_detection_label_names
from chainercv.datasets import VOCDetectionDataset
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
from SSD_for_vehicle_detection import SSD300_vd, SSD512_vd
from COWC_dataset_processed import COWC_dataset_processed, vehicle_classes
from utils import gen_dms_time_str

defaultbox_size_300 = {
    0.15: (30, 48.0, 103.5, 159, 214.5, 270, 325.5),
    0.16: (30, 48.0, 103.5, 159, 214.5, 270, 325.5),
    0.3: (24, 30, 90, 150, 210, 270, 330),
}
defaultbox_size_512 = {
    0.15: (30.72, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72),
    0.16: (30.72, 46.08, 129.02, 211.97, 294.91, 377.87, 460.8, 543.74),
    0.3: (25.6, 30.72, 116.74, 202.75, 288.79, 374.78, 460.8, 546.82),
}  # defaultbox size corresponding to the image resolution


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument(
        '--resolution', type=float, choices=(0.15,0.16,0.3), default=0.15)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    parser.add_argument('--resumemodel')
    parser.add_argument('--datadir')
    args = parser.parse_args()

    batchsize = args.batchsize
    gpu = 0
    out = args.out
    resume = args.resume

    exectime = time.time()

    global defaultbox_size_300
    global defaultbox_size_512

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

    # if args.resumemodel:
    #     serializers.load_npz(resume, model)

    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    train = TransformDataset(
        # ConcatenatedDataset(
        #     VOCDetectionDataset(year='2007', split='trainval'),
        #     VOCDetectionDataset(year='2012', split='trainval')
        # ),
        COWC_dataset_processed(split="train",datadir=args.datadir),
        Transform(model.coder, model.insize, model.mean))
    train_iter = chainer.iterators.MultiprocessIterator(train, batchsize)

    # test = VOCDetectionDataset(
    #     year='2007', split='test',
    #     use_difficult=True, return_difficult=True)
    test = COWC_dataset_processed(split="validation",datadir=args.datadir)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)

    # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

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
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(1000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(120000, 'iteration'))

    if resume:
        serializers.load_npz(resume, trainer)

    # serializers.load_npz("model/snapshot_iter_30000", trainer)
    # serializers.save_npz("model/ssd_300_0.3_30000",trainer.updater._optimizers["main"].target.model)

    trainer.run()

    exectime = time.time() - exectime
    exectime_str = gen_dms_time_str(exectime)
    print("exextime:"+exectime_str)

if __name__ == '__main__':
    main()