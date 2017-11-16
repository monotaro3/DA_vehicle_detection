# coding: utf-8

import argparse
import copy
import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers, reporter
from chainer import training, Variable
from chainer import functions as F
from chainer.training import extensions
from chainer.training import triggers
from chainer.dataset import convert

from chainercv.datasets import voc_detection_label_names
from chainercv.datasets import VOCDetectionDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms
from chainercv.evaluations import eval_detection_voc
from chainercv.utils import apply_prediction_to_iterator

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

import time

#--custom
from SSD_for_vehicle_detection import SSD300_vd, SSD512_vd
from COWC_dataset_processed import COWC_dataset_processed, vehicle_classes, Dataset_imgonly
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

def initSSD(modelname,resolution,path=None):
    if modelname == "ssd300":
        model = SSD300_vd(
            n_fg_class=len(vehicle_classes),
            pretrained_model='imagenet', defaultbox_size=defaultbox_size_300[resolution])
    elif modelname == "ssd512":
        model = SSD512_vd(
            n_fg_class=len(vehicle_classes),
            pretrained_model='imagenet', defaultbox_size=defaultbox_size_512[resolution])
    if path != None:
        serializers.load_npz(path, model)
    return model

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

class Multibox_CORAL_loss(chainer.Chain):

    def __init__(self, model, CORAL_calculation = 0,CORAL_weight=1, alpha=1, k=3):
        super( Multibox_CORAL_loss, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k
        self.CORAL_calculation = CORAL_calculation
        self.CORAL_weight = CORAL_weight
        self.extractor = self.model.extractor

    def __call__(self, s_imgs, gt_mb_locs, gt_mb_labels, t_imgs):
        # mb_locs, mb_confs = self.model(s_imgs)
        # loc_loss, conf_loss = multibox_loss(
        #     mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        # loss = loc_loss * self.alpha + conf_loss

        src_fmap = self.extractor(s_imgs)  # src feature map
        mb_locs, mb_confs = self.model.multibox(src_fmap)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        cls_loss = loc_loss * self.alpha + conf_loss  # cls loss
        batchsize_tgt = len(t_imgs)

        tgt_fmap = self.extractor(t_imgs)
        src_examples = src_fmap[0][:batchsize_tgt]
        tgt_examples = tgt_fmap[0]
        n_data, c, w, h = src_examples.shape

        if self.CORAL_calculation == 0:
            src_examples = F.transpose(src_examples, axes=(0,2,3,1))
            src_examples = F.reshape(src_examples,(n_data*w*h, c))
            tgt_examples = F.transpose(tgt_examples, axes=(0, 2, 3, 1))
            tgt_examples = F.reshape(tgt_examples, (n_data * w * h, c))
            n_data = n_data * w * h
            norm_coef = 1 / (4 * c**2)
        elif self.CORAL_calculation == 1:
            src_examples = F.im2col(src_examples,3,1,1)
            src_examples = F.reshape(src_examples,(n_data,c,3*3,w,h))
            src_examples = F.transpose(src_examples, axes=(0,3,4,1,2))
            src_examples = F.reshape(src_examples,(n_data*w*h,c*3*3))
            tgt_examples = F.im2col(tgt_examples, 3, 1, 1)
            tgt_examples = F.reshape(tgt_examples, (n_data, c, 3 * 3, w, h))
            tgt_examples = F.transpose(tgt_examples, axes=(0, 3, 4, 1, 2))
            tgt_examples = F.reshape(tgt_examples, (n_data * w * h, c * 3 * 3))
            n_data = n_data * w * h
            norm_coef = 1 / (4 * (c * 3 * 3)**2)
        else:
            src_examples = F.reshape(src_examples, (n_data,-1))
            tgt_examples = F.reshape(tgt_examples, (n_data, -1))
            norm_coef = 1 / (4 * (c * w * h) ** 2)
        xp = self.model.xp
        colvec_1 = xp.ones((1,n_data),dtype=np.float32)
        _s_tempmat = F.matmul(Variable(colvec_1),src_examples)
        _t_tempmat = F.matmul(Variable(colvec_1), tgt_examples)
        s_cov_mat = (F.matmul(F.transpose(src_examples),src_examples) - F.matmul(F.transpose(_s_tempmat),_s_tempmat) / n_data) / n_data -1 if n_data > 1 else 1
        t_cov_mat = (F.matmul(F.transpose(tgt_examples), tgt_examples) - F.matmul(F.transpose(_t_tempmat),
                                                                                  _t_tempmat) / n_data) / n_data - 1 if n_data > 1 else 1
        CORAL_loss = F.sum(F.squared_error(s_cov_mat, t_cov_mat)) * norm_coef

        loss = cls_loss + CORAL_loss * self.CORAL_weight

        chainer.reporter.report(
            {'loss': loss, 'cls_loss': cls_loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss, 'CORAL_loss_weighted': CORAL_loss * self.CORAL_weight},
            self)

        return loss

class Updater_CORAL(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        #self.loss_func = kwargs.pop('lossfunc')
        super(Updater_CORAL, self).__init__(*args, **kwargs)
        # self.t_enc = self.cls.extractor
        # self.alpha = 1
        # self.k = 3

    def update_core(self):
        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        optimizer = self.get_optimizer('main')
        xp = self.loss_func.xp

        batch_source = self.get_iterator('main').next()
        batch_source_array = convert.concat_examples(batch_source,self.device)
        batch_target = xp.array(self.get_iterator('target').next())
        batchsize = len(batch_source)

        if self.device >=0: batch_target = chainer.cuda.to_gpu(batch_target,self.device)
        #compute loss
        loss = self.loss_func(batch_source_array[0],batch_source_array[1],batch_source_array[2],batch_target)

        self.loss_func.model.cleargrads()
        loss.backward()
        optimizer.update()

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

class CORAL_evaluator(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER
    def __init__(self,src_iter,model, use_07_metric,label_names, target_eval_args,eval_doc_iter,eval_tgt_iter):
        super(CORAL_evaluator, self).__init__(
            None, model)
        #self.voc_evaluator = DetectionVOCEvaluator(**voc_args)
        from SSD_test import ssd_evaluator
        self.tgt_evaluator = ssd_evaluator(**target_eval_args)
        self.updater = target_eval_args["updater"]
        self.eval_doc_iter = eval_doc_iter
        self.eval_tgt_iter = eval_tgt_iter
        self.src_iter = src_iter
        #self.target = model
        self.use_07_metric = use_07_metric
        self.label_names = label_names

    def evaluate(self):
        target = self._targets['main']
        observation = self.tgt_evaluator.evaluate()

        #evaluate in source domain
        if self.updater.iteration % self.eval_doc_iter == 0:
            if hasattr(self.src_iter, 'reset'):
                self.src_iter.reset()
                it = self.src_iter
            else:
                it = copy.copy(self.src_iter)

            imgs, pred_values, gt_values = apply_prediction_to_iterator(
                target.predict, it)
            # delete unused iterator explicitly
            del imgs

            pred_bboxes, pred_labels, pred_scores = pred_values

            if len(gt_values) == 3:
                gt_bboxes, gt_labels, gt_difficults = gt_values
            elif len(gt_values) == 2:
                gt_bboxes, gt_labels = gt_values
                gt_difficults = None

            result = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults,
                use_07_metric=self.use_07_metric)

            report = {'s_map': result['s_map']}

            if self.label_names is not None:
                for l, label_name in enumerate(self.label_names):
                    try:
                        report['s_ap/{:s}'.format(label_name)] = result['ap'][l]
                    except IndexError:
                        report['s_ap/{:s}'.format(label_name)] = np.nan
            with reporter.report_scope(observation):
                reporter.report(report, self.target)
        return observation



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument(
        '--resolution', type=float, choices=(0.15,0.16,0.3), default=0.15)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--batchsize_tgt', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--snapshot_interval', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--iteration', type = int)
    parser.add_argument('--lrdecay_itr', type=int, nargs = 2, default = [40000,50000])
    parser.add_argument('--model_init')
    parser.add_argument('--datadir')
    parser.add_argument('--DA_data')
    parser.add_argument('--DA_valdata')
    parser.add_argument('--eval_src_itr', type = int)
    parser.add_argument('--eval_tgt_itr', type = int)
    parser.add_argument('--weightdecay', type=float,  default=0.0005)
    parser.add_argument('--opt_mode',type=str,choices = ("Adam","MomentumSGD"))
    parser.add_argument('--CORAL_calculation', type=int, choices=(0,1,2)) # 0:pixelwise, 1:patchwise, 2:fmapwise
    parser.add_argument('--CORAL_weight', type=float)

    args = parser.parse_args()

    batchsize = args.batchsize
    gpu = args.gpu
    out = args.out
    resume = args.resume

    if not (args.eval_src_itr >= args.eval_tgt_itr and args.eval_src_itr % args.eval_tgt_itr == 0):
        print('eval_src_iter must not be smaller than eval_tgt_iter and must be a multiple of eval_tgt_iter')
        exit(0)

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

    if args.model_init:
        serializers.load_npz(args.model_init, model)

    model.use_preset('evaluate')
    train_chain =  Multibox_CORAL_loss(model,args.CORAL_calculation,args.CORAL_weight)
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
    test_iter = chainer.iterators.MultiprocessIterator(
        test, batchsize, repeat=False, shuffle=False)

    target_dataset = Dataset_imgonly(args.DA_data)
    target_iter = chainer.iterators.MultiprocessIterator(target_dataset, args.batchsize_tgt)

    # initial lr is set to 1e-3 by ExponentialShift
    if args.opt_mode == "MomentumSGD":
        optimizer = chainer.optimizers.MomentumSGD()
    elif args.opt_mode == "Adam":
        optimizer = chainer.optimizers.Adam()
    optimizer.setup(train_chain)
    if args.opt_mode == "MomentumSGD":
        for param in train_chain.params():
            if param.name == 'b':
                param.update_rule.add_hook(GradientScaling(2))
            else:
                param.update_rule.add_hook(WeightDecay(args.weightdecay))

    updater_args = {
        "iterator": {'main': train_iter, 'target': target_iter, },
        "device": args.gpu
    }
    updater_args["optimizer"] = optimizer
    updater_args["loss_func"] = train_chain

    #updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    updater = Updater_CORAL(**updater_args)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out)
    if args.opt_mode == "MomentumSGD":
        trainer.extend(
            extensions.ExponentialShift('lr', 0.1, init=1e-3),
            trigger=triggers.ManualScheduleTrigger([args.lrdecay_itr[0], args.lrdecay_itr[1]], 'iteration'))

    import os
    tgt_eval_args = {"img_dir" : args.DA_valdata, "target":model,"updater":updater, "savedir":os.path.join(args.out,"bestshot"),
                     "resolution": 0.3, "modelsize": "ssd300", "evalonly": True, "label_names": vehicle_classes,
                     "save_bottom": 0.6}

    # trainer.extend(
    #     CORAL_evaluator(
    #         test_iter, model, True, vehicle_classes,tgt_eval_args,args.eval_src_itr,args.eval_tgt_itr),
    #     trigger=(args.eval_tgt_itr, 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=vehicle_classes),
        trigger=(args.eval_src_itr, 'iteration'))

    bestshot_dir = os.path.join(args.out,"bestshot")
    if not os.path.isdir(bestshot_dir): os.makedirs(bestshot_dir)

    from SSD_test import ssd_evaluator
    trainer.extend(
        ssd_evaluator(
            args.DA_valdata, model, updater, savedir=bestshot_dir, label_names=vehicle_classes),
        trigger=(args.eval_tgt_itr, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss','main/cls_loss', 'main/loss/loc', 'main/loss/conf', 'main/CORAL_loss_weighted',
         'validation/main/s_map','validation/main/map','validation/main/F1/car']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(args.iteration, 'iteration'))

    if resume:
        serializers.load_npz(resume, trainer)

    # serializers.load_npz("model/snapshot_iter_300_0.16_55000", trainer)
    # serializers.save_npz("model/ssd_300_0.16_55000",trainer.updater._optimizers["main"].target.model)

    trainer.run()

    exectime = time.time() - exectime
    exectime_str = gen_dms_time_str(exectime)
    print("exextime:"+exectime_str)

if __name__ == '__main__':
    main()