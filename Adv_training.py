import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions
from chainer import serializers
from chainercv.links.model.ssd import VGG16Extractor300
from chainer.datasets import TransformDataset

from SSD_for_vehicle_detection import *
from DA_updater import *
from COWC_dataset_processed import COWC_fmap_set, Dataset_imgonly, COWC_dataset_processed, vehicle_classes
from SSD_training import  Transform, ConcatenatedDataset
from utils import initSSD
from SSD_test import ssd_evaluator

sys.path.append(os.path.dirname(__file__))

def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/addatest', help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=10, help='Interval of evaluation')
    parser.add_argument('--display_interval', type=int, default=10, help='Interval of displaying log to console')
    # parser.add_argument('--n_dis', type=int, default=5, help='number of discriminator update per generator update')
    # parser.add_argument('--gamma', type=float, default=0.5, help='hyperparameter gamma')
    # parser.add_argument('--lam', type=float, default=10, help='gradient penalty')
    parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')
    # parser.add_argument('--output_dim', type=int, default=256, help='output dimension of the discriminator (for cramer GAN)')
    # parser.add_argument('--initencoder',  help='trained encoder which initializes target encoder')
    parser.add_argument('--dis_class', type = str, help='DA discriminator class name to be used')
    # parser.add_argument('--discriminator_init', type=str, help='discriminator file path for initialization')
    # parser.add_argument('--DA2_csize', type=int, help='channel size of conv2 of DA2_discriminator')
    # parser.add_argument('--tgt_step_init', type=int, help='initial step number of tgt training in one iteration')
    # parser.add_argument('--dis_step_init', type=int, help='initial step number of discriminator training in one iteration')
    # parser.add_argument('--tgt_step_schedule', type=int, nargs = "*", help='schedule of step number of tgt training in one iteration')
    # parser.add_argument('--Alt_update_param', type=int, help='parameters of alternative update', nargs = 3, choices = [0,1])
    # parser.add_argument('--multibatch_times', type=int, help='number of multiplication of batchsize for discriminator learning')
    parser.add_argument('--updater', type=str, help='Updater class name to be used')
    parser.add_argument('--reconstructor', type=str, choices = ["deconv","unpool","unpool_conv"], help='upsampling type of reconstructor')
    parser.add_argument('--rec_weight', type=float, default=1.)
    parser.add_argument('--rec_batch_split', type=int, default=16)
    parser.add_argument('--rec_noalt', action="store_true")
    parser.add_argument('--rec_noadv', action="store_true")
    parser.add_argument('--source_dataset', type=str, default= "E:/work/vehicle_detection_dataset/cowc_300px_0.3_fmap" , help='source dataset directory')
    # parser.add_argument('--fixed_source_dataset', type=str, help='source fmap dataset directory')
    parser.add_argument('--target_dataset', type=str, default= "E:/work/vehicle_detection_dataset/Khartoum_adda" , help='target dataset directory')
    # parser.add_argument('--mode', type=str, choices = ["DA1", "DA1_buf","DA1_buf_multibatch","DA_fix_dis"] ,default="DA1", help='mode of domain adaptation')
    parser.add_argument('--ssdpath', type=str,  help='SSD model file')
    parser.add_argument('--evalimg', type=str, help='img path for evaluation')
    parser.add_argument('--resume', type=str, help='trainer snapshot path for resume')
    parser.add_argument('--bufsize', type=int, help='size of buffer for discriminator training')
    parser.add_argument('--bufmode', type=int, default=1, help='mode of buffer(0:align src and tgt, 1:not align, 2:sort by loss value)')
    parser.add_argument('--tgt_anno_data', type=str, help='target anotation dataset directory')
    parser.add_argument('--s_t_ratio',type=int, nargs=2)
    parser.add_argument('--out_progress')

    args = parser.parse_args()

    report_keys = ["loss_cls", "loss_t_enc", "loss_dis", 'loss_dis_src', 'loss_dis_tgt', 'validation/main/map',
                   'validation/main/RR/car',
                   'validation/main/PR/car', 'validation/main/FAR/car', 'validation/main/F1/car', 'lr_dis', 'lr_cls']
    if args.reconstructor:
        report_keys += ["loss_rec"]

    Discriminator = eval(args.dis_class)
    discriminator = Discriminator()
    ssd_model = initSSD("ssd300",0.3,args.ssdpath)
    models = [discriminator, ssd_model]
    if args.reconstructor:
        from SSD_for_vehicle_detection import Recontructor
        reconstructor = Recontructor(args.reconstructor)
        models.append(reconstructor)

    if args.tgt_anno_data:
        if args.s_t_ratio:
            if args.s_t_ratio[0] == 0:
                source_dataset = TransformDataset(
                    COWC_dataset_processed(split="train", datadir=args.tgt_anno_data),
                    Transform(ssd_model.coder, ssd_model.insize, ssd_model.mean))
            else:
                s_batchsize = int(args.batchsize * args.s_t_ratio[0] / sum(args.s_t_ratio))
                t_batchsize = args.batchsize - s_batchsize
                if s_batchsize <= 0:
                    print("invalid batchsize")
                    exit(0)
                source_dataset = TransformDataset(
                    COWC_dataset_processed(split="train", datadir=args.source_dataset),
                    Transform(ssd_model.coder, ssd_model.insize, ssd_model.mean))
                t_train = TransformDataset(
                    COWC_dataset_processed(split="train", datadir=args.tgt_anno_data),
                    Transform(ssd_model.coder, ssd_model.insize, ssd_model.mean))
                #train_iter1 = chainer.iterators.MultiprocessIterator(s_train, s_batchsize)
                train_iter_target = chainer.iterators.MultiprocessIterator(t_train, t_batchsize)
        else:
            source_dataset = TransformDataset(
                ConcatenatedDataset(
                    COWC_dataset_processed(split="train", datadir=args.source_dataset),
                    COWC_dataset_processed(split="train", datadir=args.tgt_anno_data)
                ),
                # COWC_dataset_processed(split="train", datadir=args.datadir),
                Transform(ssd_model.coder, ssd_model.insize, ssd_model.mean))
            #train_iter1 = chainer.iterators.MultiprocessIterator(s_train, args.batchsize)
    else:
        source_dataset = TransformDataset(
            COWC_dataset_processed(split="train", datadir=args.source_dataset),
            Transform(ssd_model.coder, ssd_model.insize, ssd_model.mean))
    target_dataset = Dataset_imgonly(args.target_dataset)

    train_iter1 = chainer.iterators.MultiprocessIterator(source_dataset, args.batchsize)
    train_iter2 = chainer.iterators.MultiprocessIterator(target_dataset, args.batchsize)

    updater_args = {
        "iterator": {'main': train_iter1, 'target': train_iter2, },
        "device": args.gpu
    }

    if args.tgt_anno_data and args.s_t_ratio and args.s_t_ratio[0] != 0:
        updater_args['iterator']['tgt_annotation']= train_iter_target

    Updater = eval(args.updater)

    if args.bufsize < int(args.batchsize/2):
        print("bufsize must not be smaller than batchsize/2")
        raise ValueError
    buffer = fmapBuffer(args.bufsize,mode=args.bufmode,discriminator=discriminator,gpu=args.gpu)
    updater_args["buffer"] = buffer

    if args.reconstructor:
        updater_args["rec_weight"] = args.rec_weight
        updater_args["rec_batch_split"] = args.rec_batch_split
        updater_args["rec_noalt"] = args.rec_noalt
        updater_args["rec_noadv"] = args.rec_noadv

    # Set up optimizers
    opts = {}
    if args.reconstructor and not args.rec_noadv:
        opts["opt_dis"] = make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    opts["opt_cls"] = make_optimizer(ssd_model, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    if args.reconstructor:
        opts["opt_rec"] = make_optimizer(reconstructor, args.adam_alpha, args.adam_beta1, args.adam_beta2)

    updater_args["optimizer"] = opts
    updater_args["models"] = models

    # Set up updater and trainer
    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    trainer.extend(extensions.observe_lr(optimizer_name="opt_cls", observation_key='lr_cls'),
                   trigger=(args.display_interval, 'iteration'))
    trainer.extend(extensions.snapshot(), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(args.display_interval, 'iteration')))

    printreport_args = {"entries": report_keys}
    progress_args = {"update_interval": 10}
    if args.out_progress:
        fo = open(args.out_progress, 'w')
        printreport_args["out"] = fo
        progress_args["out"] = fo

    trainer.extend(extensions.PrintReport(**printreport_args),
                   trigger=(args.display_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(**progress_args))

    trainer.extend(
        extensions.snapshot_object(ssd_model, 'ssdmodel_iter_{.updater.iteration}'),
        trigger=(args.max_iter, 'iteration'))
    if args.reconstructor:
        trainer.extend(
            extensions.snapshot_object(reconstructor, 'reconstructor_iter_{.updater.iteration}'),
            trigger=(args.max_iter, 'iteration'))

    bestshot_dir = os.path.join(args.out,"bestshot")
    if not os.path.isdir(bestshot_dir): os.makedirs(bestshot_dir)

    trainer.extend(
        ssd_evaluator(
            args.evalimg, ssd_model,updater,savedir=bestshot_dir, label_names=vehicle_classes),
        trigger=(args.evaluation_interval, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    if args.out_progress:
        fo.close()

if __name__ == '__main__':
    main()