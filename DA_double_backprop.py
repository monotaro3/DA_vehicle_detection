import argparse
from utils import initSSD
from COWC_dataset_processed import Dataset_imgonly, COWC_dataset_processed, vehicle_classes
from SSD_training import Transform
from chainer.datasets import TransformDataset
import chainer
from chainercv.links.model.ssd import GradientScaling, multibox_loss
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

def recursive_transfer_grad_var(layer_s, layer_t,dst,lr):
    #Source and target models (roots of layer_s and layer_t) must have exactly the same structure.
    if len(layer_s._params) > 0:
        for p in layer_s._params:
            _grad_var = layer_s.__dict__[p].grad_var
            if _grad_var is not None:
                grad_var = F.copy(_grad_var,dst)
                print(grad_var)
                with cupy.cuda.Device(dst):
                    layer_t.__dict__[p] += grad_var * lr
    if '_children' in layer_s.__dict__:
        if len(layer_s._children) > 0:
            for c in layer_s._children:
                if isinstance(c,str):
                    recursive_transfer_grad_var(layer_s.__dict__[c],layer_t.__dict__[c],dst,lr)
                else:
                    recursive_transfer_grad_var(c,layer_t._children[layer_s._children.index(c)],dst,lr)

# def recursive_transfer_grad_var(model_source, model_target,dst,lr):
#     #model_source and model_target must have exactly the same structure
#     _recursive_transfer_grad_var(model_source, model_target,dst,lr)

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

        #for parameter initialization
        if self.iteration == 0:
            with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
                ssd_predict_variable(self.model_2, batch_unlabeled, raw=True)
                ssd_predict_variable(self.model_3, batch_unlabeled, raw=True)
                ssd_predict_variable(self.model_4, batch_unlabeled, raw=True)

        self.model_1.cleargrads()
        self.model_2.cleargrads()
        self.model_3.cleargrads()
        self.model_4.cleargrads()

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

        self.model_4.addgrads(self.model_2)
        opt_model_4.update()

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
    parser.add_argument('--lr', type=float, default=1e-3)
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
    parser.add_argument('--lrdecay_schedule', nargs = '*', type=int, default=[80000,100000])
    parser.add_argument('--loss_mode', type=int, default=0, choices = [0,1])
    parser.add_argument('--eval_tgt_itr', type=int,default=100)

    args = parser.parse_args()

    model_1 = initSSD(args.model, args.resolution, args.model_init_1)
    model_2 = initSSD(args.model, args.resolution, args.model_init_2)
    model_3 = initSSD(args.model, args.resolution, args.model_init_2)
    model_4 = initSSD(args.model, args.resolution, args.model_init_2)
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
        optimizer1 = chainer.optimizers.MomentumSGD(args.lr)
        optimizer4 = chainer.optimizers.MomentumSGD(args.lr)
        optimizer1.setup(model_1)
        optimizer4.setup(model_4)
        for model in (model_1, model_4):
            for param in model.params():
                if param.name == 'b':
                    param.update_rule.add_hook(GradientScaling(2))
                else:
                    param.update_rule.add_hook(WeightDecay(args.weightdecay))
    else:
        optimizer1 = chainer.optimizers.Adam(alpha=args.lr)
        optimizer4 = chainer.optimizers.Adam(alpha=args.lr)
        optimizer1.setup(model_1)
        optimizer4.setup(model_4)

    updater_args = {
        "iterator": {'main': train_iter1, 'target': train_iter2, },
        'optimizer': {'opt_model_1':optimizer1, 'opt_model_4':optimizer4,},
        # "device": args.gpu
        'loss_mode' : args.loss_mode,
        'models': (model_1, model_2, model_3, model_4),
        'lr' : args.lr,
    }

    updater = Updater_dbp(**updater_args)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    # trainer.extend(
    #     extensions.ExponentialShift('lr', 0.1, init=args.lr),
    #     trigger=triggers.ManualScheduleTrigger(list(args.lrdecay_schedule), 'iteration'))

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
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss_model2', 'main/loss_model2/loc', 'main/loss_model2/conf',
         'main/loss_model3', 'main/loss_model3/loc', 'main/loss_model3/conf',
         'validation/main/map']),
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

    trainer.run()

if __name__ == '__main__':
    main()