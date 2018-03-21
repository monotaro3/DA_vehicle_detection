import argparse
from utils import initSSD
from COWC_dataset_processed import Dataset_imgonly, COWC_dataset_processed, vehicle_classes
from SSD_training import Transform
from chainer.datasets import TransformDataset
import chainer
from chainercv.links.model.ssd import GradientScaling #, multibox_loss
from chainer.optimizer import WeightDecay
from chainer import training, serializers
from chainer.training import extensions
from chainer.training import triggers
from chainercv.extensions import DetectionVOCEvaluator
from chainer.dataset import convert
from SSD_for_vehicle_detection import ssd_predict_variable, resize_bbox_variable, multibox_encode_variable
import chainer.functions as F
import os
from SSD_test import ssd_evaluator
import cupy
import numpy as np

def _elementwise_softmax_cross_entropy(x, t):
    assert x.shape[:-1] == t.shape
    shape = t.shape
    x = F.reshape(x, (-1, x.shape[-1]))
    t = F.flatten(t)
    return F.reshape(
        F.softmax_cross_entropy(x, t, reduce='no',enable_double_backprop=True), shape)


def _hard_negative(x, positive, k):
    rank = (x * (positive - 1)).argsort(axis=1).argsort(axis=1)
    hard_negative = rank < (positive.sum(axis=1) * k)[:, np.newaxis]
    return hard_negative


def multibox_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k):
    """Computes multibox losses.

    This is a loss function used in [#]_.
    This function returns :obj:`loc_loss` and :obj:`conf_loss`.
    :obj:`loc_loss` is a loss for localization and
    :obj:`conf_loss` is a loss for classification.
    The formulas of these losses can be found in
    the equation (2) and (3) in the original paper.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        mb_locs (chainer.Variable or array): The offsets and scales
            for predicted bounding boxes.
            Its shape is :math:`(B, K, 4)`,
            where :math:`B` is the number of samples in the batch and
            :math:`K` is the number of default bounding boxes.
        mb_confs (chainer.Variable or array): The classes of predicted
            bounding boxes.
            Its shape is :math:`(B, K, n\_class)`.
            This function assumes the first class is background (negative).
        gt_mb_locs (chainer.Variable or array): The offsets and scales
            for ground truth bounding boxes.
            Its shape is :math:`(B, K, 4)`.
        gt_mb_labels (chainer.Variable or array): The classes of ground truth
            bounding boxes.
            Its shape is :math:`(B, K)`.
        k (float): A coefficient which is used for hard negative mining.
            This value determines the ratio between the number of positives
            and that of mined negatives. The value used in the original paper
            is :obj:`3`.

    Returns:
        tuple of chainer.Variable:
        This function returns two :obj:`chainer.Variable`: :obj:`loc_loss` and
        :obj:`conf_loss`.
    """
    mb_locs = chainer.as_variable(mb_locs)
    mb_confs = chainer.as_variable(mb_confs)
    gt_mb_locs = chainer.as_variable(gt_mb_locs)
    gt_mb_labels = chainer.as_variable(gt_mb_labels)

    xp = chainer.cuda.get_array_module(gt_mb_labels.array)

    positive = gt_mb_labels.array > 0
    n_positive = positive.sum()
    if n_positive == 0:
        z = chainer.Variable(xp.zeros((), dtype=np.float32))
        return z, z

    loc_loss = F.huber_loss(mb_locs, gt_mb_locs, 1, reduce='no')
    loc_loss = F.sum(loc_loss, axis=-1)
    loc_loss *= positive.astype(loc_loss.dtype)
    loc_loss = F.sum(loc_loss) / n_positive

    conf_loss = _elementwise_softmax_cross_entropy(mb_confs, gt_mb_labels)
    hard_negative = _hard_negative(conf_loss.array, positive, k)
    conf_loss *= xp.logical_or(positive, hard_negative).astype(conf_loss.dtype)
    conf_loss = F.sum(conf_loss) / n_positive

    return loc_loss, conf_loss

def recursive_transfer_grad_var(layer_s, layer_t,dst,lr):
    #Source and target models (roots of layer_s and layer_t) must have exactly the same structure.
    if len(layer_s._params) > 0:
        for p in layer_s._params:
            _grad_var = layer_s.__dict__[p].grad_var
            if _grad_var is not None:
                grad_var = F.copy(_grad_var,dst)
                # print(grad_var)
                with cupy.cuda.Device(dst):
                    layer_t.__dict__[p] += grad_var * lr
    if '_children' in layer_s.__dict__:
        if len(layer_s._children) > 0:
            for c in layer_s._children:
                if isinstance(c,str):
                    recursive_transfer_grad_var(layer_s.__dict__[c],layer_t.__dict__[c],dst,lr)
                else:
                    recursive_transfer_grad_var(c,layer_t._children[layer_s._children.index(c)],dst,lr)

class Updater_dbp_sgpu(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.model_1, self.model_2, self.model_3, self.model_4 =  kwargs.pop('models')
        self.loss_mode = kwargs.pop('loss_mode')
        self.lr1 =  kwargs.pop('lr1')
        self.lr2 = kwargs.pop('lr2')
        self.constraint = kwargs.pop('constraint')
        super(Updater_dbp_sgpu, self).__init__(*args, **kwargs)
        self.alpha = 1
        self.k = 3

    def update_core(self):
        opt_model_1 = self.get_optimizer('opt_model_1')
        opt_model_4 = self.get_optimizer('opt_model_4')

        batch_labeled = self.get_iterator('main').next()
        batch_unlabeled = self.get_iterator('target').next()
        batchsize = len(batch_labeled)

        if self.constraint:
            batch_labeled_array = convert.concat_examples(batch_labeled, 0)
            mb_locs, mb_confs = self.model_1(batch_labeled_array[0])
            loc_loss, conf_loss = multibox_loss(
                mb_locs, mb_confs, batch_labeled_array[1], batch_labeled_array[2], self.k)
            loss_model_1 = loc_loss * self.alpha + conf_loss  # cls loss

            chainer.reporter.report(
                {'loss_model1': loss_model_1, 'loss_model1/loc': loc_loss, 'loss_model1/conf': conf_loss})
            self.model_1.cleargrads()
            loss_model_1.backward()
            loss_model_1.unchain_backward()

        if self.loss_mode == 0:
            bboxes, labels, scores = ssd_predict_variable(self.model_1, batch_unlabeled)
            mb_locs_l = []
            mb_labels_l = []
            for bbox, label in zip(bboxes, labels):
                mb_loc, mb_label = multibox_encode_variable(self.model_1.coder, bbox, label)
                mb_locs_l.append(mb_loc)
                mb_labels_l.append(mb_label)
            mb_locs_l = F.stack(mb_locs_l)
            mb_labels_l = F.stack(mb_labels_l)
        else:
            mb_locs_l, mb_confs_l = ssd_predict_variable(self.model_1, batch_unlabeled, raw=True)
            mb_labels_l = F.argmax(mb_confs_l,axis=2)

        if not self.constraint:
            self.model_1.cleargrads()

        mb_locs, mb_confs = ssd_predict_variable(self.model_2, batch_unlabeled, raw=True)

        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, mb_locs_l, mb_labels_l, self.k)
        loss_model_2 = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss_model2': loss_model_2, 'loss_model2/loc': loc_loss, 'loss_model2/conf': conf_loss})

        self.model_2.cleargrads()

        if self.iteration == 0:
            self.model_3.copyparams(self.model_2)
            self.model_4.copyparams(self.model_2)

        params = []
        param_generator = self.model_2.params()
        for param in param_generator:
            params.append(param)

        # with chainer.cuda.get_device(mb_locs_l_g1.data):
        gradients = chainer.grad([loss_model_2], params,set_grad=True, enable_double_backprop=True)

        recursive_transfer_grad_var(self.model_2, self.model_3,dst=0, lr=self.lr2)

        batch_labeled_array = convert.concat_examples(batch_labeled, 0)
        # with chainer.cuda.get_device(batch_labeled_array[0]):
        mb_locs, mb_confs = self.model_3(batch_labeled_array[0])
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, batch_labeled_array[1], batch_labeled_array[2], self.k)
        loss_model_3 = loc_loss * self.alpha + conf_loss  # cls loss

        chainer.reporter.report(
            {'loss_model3': loss_model_3, 'loss_model3/loc': loc_loss, 'loss_model3/conf': conf_loss})

        self.model_4.cleargrads()
        self.model_4.addgrads(self.model_2)
        opt_model_4.update()

        self.model_3.cleargrads()
        loss_model_3.backward()
        opt_model_1.update()

        loss_model_2.unchain_backward()
        loss_model_3.unchain_backward()

        self.model_2.copyparams(self.model_4)
        self.model_3.copyparams(self.model_4)


class Updater_trainSSD(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.model_1, self.model_2, self.model_3, self.model_4 = kwargs.pop('models')
        self.loss_mode = kwargs.pop('loss_mode')
        self.lr1 = kwargs.pop('lr1')
        self.lr2 = kwargs.pop('lr2')
        self.constraint = kwargs.pop('constraint')
        super(Updater_trainSSD, self).__init__(*args, **kwargs)
        self.alpha = 1
        self.k = 3

    def update_core(self):
        opt_model_2 = self.get_optimizer('opt_model_1')
        # opt_model_4 = self.get_optimizer('opt_model_4')

        batch_labeled = self.get_iterator('main').next()
        batch_unlabeled = self.get_iterator('target').next()
        batchsize = len(batch_labeled)

        if self.loss_mode == 0:
            # bboxes, labels, scores = ssd_predict_variable(self.model_1, batch_unlabeled)
            # mb_locs_l = []
            # mb_labels_l = []
            # for bbox, label in zip(bboxes, labels):
            #     mb_loc, mb_label = multibox_encode_variable(self.model_1.coder, bbox, label)
            #     mb_locs_l.append(mb_loc)
            #     mb_labels_l.append(mb_label)
            # mb_locs_l = F.stack(mb_locs_l)
            # mb_labels_l = F.stack(mb_labels_l)
            bboxes, labels, scores = self.model_1.predict(batch_unlabeled)
            mb_locs_l = []
            mb_labels_l = []
            xp = self.model_1.xp
            for bbox, label in zip(bboxes, labels):
                mb_loc, mb_label = self.model_1.coder.encode(xp.asarray(bbox),xp.asarray(label))
                mb_locs_l.append(mb_loc)
                mb_labels_l.append(mb_label)

            mb_locs_l = xp.stack(mb_locs_l)
            mb_labels_l = xp.stack(mb_labels_l)
        else:
            with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
                mb_locs_l, mb_confs_l = ssd_predict_variable(self.model_1, batch_unlabeled, raw=True)
                mb_labels_l = F.argmax(mb_confs_l, axis=2)

        self.model_1.cleargrads()

        mb_locs, mb_confs = ssd_predict_variable(self.model_2, batch_unlabeled, raw=True)

        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, mb_locs_l, mb_labels_l, self.k)
        loss_model_2 = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss_model2': loss_model_2, 'loss_model2/loc': loc_loss, 'loss_model2/conf': conf_loss})

        self.model_2.cleargrads()
        loss_model_2.backward()
        opt_model_2.update()

class Updater_dbp(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.model_1, self.model_2, self.model_3, self.model_4 =  kwargs.pop('models')
        self.loss_mode = kwargs.pop('loss_mode')
        self.lr =  kwargs.pop('lr')
        super(Updater_dbp, self).__init__(*args, **kwargs)
        self.alpha = 1
        self.k = 3

    def update_core(self):
        opt_model_1 = self.get_optimizer('opt_model_1')
        opt_model_4 = self.get_optimizer('opt_model_4')

        batch_labeled = self.get_iterator('main').next()
        batch_unlabeled = self.get_iterator('target').next()
        batchsize = len(batch_labeled)

        # #for parameter initialization
        # if self.iteration == 0:
        #     self.model_2.predict(batch_unlabeled)
        #     self.model_3.predict(batch_unlabeled)
        #     self.model_4.predict(batch_unlabeled)

        # self.model_1.cleargrads()
        # self.model_2.cleargrads()
        # self.model_3.cleargrads()
        # self.model_4.cleargrads()

        # batch_unlabeled_array_g0 = convert.concat_examples(batch_unlabeled, 0)
        if self.loss_mode == 0:
            bboxes, labels, scores = ssd_predict_variable(self.model_1, batch_unlabeled)
            mb_locs_l = []
            mb_labels_l = []
            for bbox, label in zip(bboxes, labels):
                mb_loc, mb_label = multibox_encode_variable(self.model_1.coder, bbox, label)
                mb_locs_l.append(mb_loc)
                mb_labels_l.append(mb_label)
            mb_locs_l = F.stack(mb_locs_l)
            mb_labels_l = F.stack(mb_labels_l)
        else:
            # x = list()
            # sizes = list()
            # for img in batch_unlabeled:
            #     _, H, W = img.shape
            #     img = self.model_1._prepare(img)
            #     x.append(self.model_1.xp.array(img))
            #     sizes.append((H, W))
            #
            # x = chainer.Variable(self.model_1.xp.stack(x))
            # mb_locs_l, mb_labels_l = self.model_1(x)
            mb_locs_l, mb_labels_l = ssd_predict_variable(self.model_1, batch_unlabeled, raw=True)

        self.model_1.cleargrads()

        mb_locs_l_g1 = F.copy(mb_locs_l,dst=1)
        mb_labels_l_g1 = F.copy(mb_labels_l,dst=1)

        # x = list()
        # sizes = list()
        # for img in batch_unlabeled:
        #     _, H, W = img.shape
        #     img = self.model_2._prepare(img)
        #     x.append(self.model_2.xp.array(img))
        #     sizes.append((H, W))
        #
        # x = chainer.Variable(self.model_2.xp.stack(x))
        # mb_locs, mb_confs = self.model_2(x)
        with chainer.cuda.get_device(mb_locs_l_g1.data):
            mb_locs, mb_confs = ssd_predict_variable(self.model_2, batch_unlabeled, raw=True)

            loc_loss, conf_loss = multibox_loss(
                mb_locs, mb_confs, mb_locs_l_g1, mb_labels_l_g1, self.k)
            loss_model_2 = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss_model2': loss_model_2, 'loss_model2/loc': loc_loss, 'loss_model2/conf': conf_loss})

        self.model_2.cleargrads()

        if self.iteration == 0:
            self.model_3.copyparams(self.model_2)
            self.model_4.copyparams(self.model_2)

        params = []
        param_generator = self.model_2.params()
        for param in param_generator:
            params.append(param)

        with chainer.cuda.get_device(mb_locs_l_g1.data):
            gradients = chainer.grad([loss_model_2], params,set_grad=True, enable_double_backprop=True)

        recursive_transfer_grad_var(self.model_2, self.model_3,dst=2, lr=self.lr)

        batch_labeled_array = convert.concat_examples(batch_labeled, 2)
        with chainer.cuda.get_device(batch_labeled_array[0]):
            mb_locs, mb_confs = self.model_3(batch_labeled_array[0])
            loc_loss, conf_loss = multibox_loss(
                mb_locs, mb_confs, batch_labeled_array[1], batch_labeled_array[2], self.k)
            loss_model_3 = loc_loss * self.alpha + conf_loss  # cls loss

        chainer.reporter.report(
            {'loss_model3': loss_model_3, 'loss_model3/loc': loc_loss, 'loss_model3/conf': conf_loss})

        self.model_4.cleargrads()
        self.model_4.addgrads(self.model_2)
        opt_model_4.update()

        self.model_3.cleargrads()
        loss_model_3.backward()
        opt_model_1.update()

        loss_model_2.unchain_backward()
        loss_model_3.unchain_backward()

        self.model_2.copyparams(self.model_4)
        self.model_3.copyparams(self.model_4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument(
        '--resolution', type=float, choices=(0.15,0.16,0.3), default=0.3)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr1', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=1e-3)
    parser.add_argument('--iteration', type=int, default=120000)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    parser.add_argument('--optimizer',type=str,default='Momentum_SGD',choices=['Momentum_SGD','Adam'])
    parser.add_argument('--model_init_1', type=str, help='model1 to be loaded')
    parser.add_argument('--model_init_2', type=str, help='model2 to be loaded')
    parser.add_argument('--dataset_labeled', type=str, help='data with labels')
    parser.add_argument('--dataset_unlabeled', type=str, help='data without labels')
    parser.add_argument('--target_valdata', type=str, help='dir of labeled target images for validation')
    parser.add_argument('--weightdecay', type=float,  default=0.0005)
    parser.add_argument('--lrdecay_schedule', nargs = '*', type=int, help='iteration number(s) of lr decay such as \'80000 100000\'')
    parser.add_argument('--loss_mode', type=int, default=0, choices = [0,1])
    parser.add_argument('--eval_tgt_itr', type=int,default=100)
    parser.add_argument("--single_gpu", action='store_true')
    parser.add_argument("--constraint", action='store_true')
    parser.add_argument("--ssd_pretrain", action='store_true')

    args = parser.parse_args()

    model_1 = initSSD(args.model, args.resolution, args.model_init_1)
    model_2 = initSSD(args.model, args.resolution, args.model_init_2)
    model_3 = initSSD(args.model, args.resolution, args.model_init_2)
    model_4 = initSSD(args.model, args.resolution, args.model_init_2)
    if args.single_gpu:
        model_1.to_gpu(0)
        model_2.to_gpu(0)
        model_3.to_gpu(0)
        model_4.to_gpu(0)
    else:
        model_1.to_gpu(0)
        model_2.to_gpu(1)
        model_3.to_gpu(2)
        model_4.to_gpu(3)

    dataset_labeled = TransformDataset(COWC_dataset_processed(split="train", datadir=args.dataset_labeled),
                        Transform(model_1.coder, model_1.insize, model_1.mean))
    dataset_unlabeled = Dataset_imgonly(args.dataset_unlabeled)

    train_iter1 = chainer.iterators.MultiprocessIterator(dataset_labeled, args.batchsize)
    train_iter2 = chainer.iterators.MultiprocessIterator(dataset_unlabeled, args.batchsize)

    test = COWC_dataset_processed(split="validation", datadir=args.dataset_labeled)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    if args.optimizer == 'Momentum_SGD':
        optimizer1 = chainer.optimizers.MomentumSGD(args.lr1)
        optimizer4 = chainer.optimizers.MomentumSGD(args.lr2)
    else:
        optimizer1 = chainer.optimizers.Adam(alpha=args.lr1)
        optimizer4 = chainer.optimizers.Adam(alpha=args.lr2)

    if args.ssd_pretrain:
        optimizer1.setup(model_2)
    else:
        optimizer1.setup(model_1)
    optimizer4.setup(model_4)

    if args.ssd_pretrain:
        optimizer_args = {'opt_model_1': optimizer1}
    else:
        optimizer_args = {'opt_model_1':optimizer1, 'opt_model_4':optimizer4,}

    if args.optimizer == 'Momentum_SGD':
        for model in (optimizer1.target, optimizer4.target):
            for param in model.params():
                if param.name == 'b':
                    param.update_rule.add_hook(GradientScaling(2))
                else:
                    param.update_rule.add_hook(WeightDecay(args.weightdecay))

    updater_args = {
        "iterator": {'main': train_iter1, 'target': train_iter2, },
        'optimizer': optimizer_args,
        # "device": args.gpu
        'loss_mode' : args.loss_mode,
        'models': (model_1, model_2, model_3, model_4),
        'lr1': args.lr1,
        'lr2': args.lr2,
        'constraint': args.constraint,
    }

    if args.single_gpu:
        updater = Updater_dbp_sgpu(**updater_args)
    else:
        updater = Updater_dbp(**updater_args)
    if args.ssd_pretrain:
        updater = Updater_trainSSD(**updater_args)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    if args.lrdecay_schedule:
        decay_attribute = 'lr' if args.optimizer =='Momentum_SGD' else 'alpha'
        trainer.extend(
            extensions.ExponentialShift(decay_attribute, 0.1, init=args.lr1,optimizer=optimizer1),
            trigger=triggers.ManualScheduleTrigger(list(args.lrdecay_schedule), 'iteration'))
        trainer.extend(
            extensions.ExponentialShift(decay_attribute, 0.1, init=args.lr2, optimizer=optimizer4),
            trigger=triggers.ManualScheduleTrigger(list(args.lrdecay_schedule), 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model_2, use_07_metric=True,
            label_names=vehicle_classes),
        trigger=(1000, 'iteration'))

    bestshot_dir = os.path.join(args.out, "bestshot")
    if not os.path.isdir(bestshot_dir): os.makedirs(bestshot_dir)

    trainer.extend(
        ssd_evaluator(
            args.target_valdata, model_1, updater, savedir=bestshot_dir, label_names=vehicle_classes),
        trigger=(args.eval_tgt_itr, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr('opt_model_1'), trigger=log_interval)

    log_key = ['epoch', 'iteration', 'lr',
         'main/loss_model2', 'main/loss_model2/loc', 'main/loss_model2/conf',
         'main/loss_model3', 'main/loss_model3/loc', 'main/loss_model3/conf',
         'validation/main/map', 'validation/main/RR/car',
         'validation/main/PR/car', 'validation/main/FAR/car', 'validation/main/F1/car'
         ]

    if args.constraint:
        log_key += ['main/loss_model1', 'main/loss_model1/loc', 'main/loss_model1/conf',]

    trainer.extend(extensions.PrintReport(
        log_key),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(1000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model_1, 'model1_iter_{.updater.iteration}'),
        trigger=(args.iteration, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model_2, 'model2_iter_{.updater.iteration}'),
        trigger=(args.iteration, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    # serializers.save_npz(
    #     os.path.join(args.out, "model_2_iter_7000"), trainer.updater._optimizers["opt_model_1"].target)

    trainer.run()

if __name__ == '__main__':
    main()