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

# from common.dataset import Cifar10Dataset
# from common.evaluation import sample_generate, sample_generate_light, calc_inception, calc_FID
# from common.record import record_setting
# import common.net

def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train script')
    #parser.add_argument('--algorithm', '-a', type=str, default="dcgan", help='GAN algorithm')
    #parser.add_argument('--architecture', type=str, default="dcgan", help='Network architecture')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/addatest', help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=10, help='Interval of evaluation')
    parser.add_argument('--display_interval', type=int, default=10, help='Interval of displaying log to console')
    parser.add_argument('--n_dis', type=int, default=5, help='number of discriminator update per generator update')
    parser.add_argument('--gamma', type=float, default=0.5, help='hyperparameter gamma')
    parser.add_argument('--lam', type=float, default=10, help='gradient penalty')
    parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')
    parser.add_argument('--output_dim', type=int, default=256, help='output dimension of the discriminator (for cramer GAN)')
    parser.add_argument('--initencoder',  help='trained encoder which initializes target encoder')
    parser.add_argument('--DA_model', type = str, help='DA discriminator class name to be used')
    parser.add_argument('--discriminator_init', type=str, help='discriminator file path for initialization')
    parser.add_argument('--DA2_csize', type=int, help='channel size of conv2 of DA2_discriminator')
    parser.add_argument('--tgt_step_init', type=int, help='initial step number of tgt training in one iteration')
    parser.add_argument('--dis_step_init', type=int, help='initial step number of discriminator training in one iteration')
    parser.add_argument('--tgt_step_schedule', type=int, nargs = "*", help='schedule of step number of tgt training in one iteration')
    parser.add_argument('--Alt_update_param', type=int, help='parameters of alternative update', nargs = 3, choices = [0,1])
    parser.add_argument('--multibatch_times', type=int, help='number of multiplication of batchsize for discriminator learning')
    parser.add_argument('--updater', type=str, help='Updater class name to be used')
    parser.add_argument('--source_dataset', type=str, default= "E:/work/vehicle_detection_dataset/cowc_300px_0.3_fmap" , help='source dataset directory')
    parser.add_argument('--fixed_source_dataset', type=str, help='source fmap dataset directory')
    parser.add_argument('--target_dataset', type=str, default= "E:/work/vehicle_detection_dataset/Khartoum_adda" , help='target dataset directory')
    parser.add_argument('--mode', type=str, choices = ["DA1", "DA1_buf","DA1_buf_multibatch","DA_fix_dis"] ,default="DA1", help='mode of domain adaptation')
    parser.add_argument('--ssdpath', type=str,  help='SSD model file')
    parser.add_argument('--evalimg', type=str, help='img path for evaluation')
    parser.add_argument('--resume', type=str, help='trainer snapshot path for resume')
    parser.add_argument('--bufsize', type=int, help='size of buffer for discriminator training')
    parser.add_argument('--bufmode', type=int, help='mode of buffer(0:align src and tgt, 1:not align, 2:sort by loss value)')
    parser.add_argument('--tgt_anno_data', type=str, help='target anotation dataset directory')
    parser.add_argument('--s_t_ratio',type=int, nargs=2)
    parser.add_argument('--out_progress')

    args = parser.parse_args()

    if args.mode in ["DA1","DA1_buf","DA1_buf_multibatch","DA_fix_dis"]:
        report_keys = ["loss_cls","loss_t_enc", "loss_dis",'loss_dis_src','loss_dis_tgt', 'validation/main/map','validation/main/RR/car',
                       'validation/main/PR/car','validation/main/FAR/car','validation/main/F1/car','lr_dis','lr_cls']
    else:
        report_keys = ["loss_t_enc", "loss_dis"]

        # Set up dataset

        source_dataset = COWC_fmap_set(args.source_dataset)
        target_dataset = Dataset_imgonly(args.target_dataset)

    # train_iter1 = chainer.iterators.SerialIterator(source_dataset, args.batchsize)
    # train_iter2 = chainer.iterators.SerialIterator(target_dataset, args.batchsize)


    # Setup algorithm specific networks and updaters
    models = []
    opts = {}


    # if args.algorithm == "dcgan":
    #     from dcgan.updater import Updater
    #     if args.architecture=="dcgan":
    #         generator = common.net.DCGANGenerator()
    #         discriminator = common.net.DCGANDiscriminator()
    #     else:
    #         raise NotImplementedError()
    s_batchsize = args.batchsize

    if args.mode in  ["DA1", "DA1_buf","DA1_buf_multibatch","DA_fix_dis"]:
        Updater = DA_updater1
        if args.DA_model:
            Discriminator = eval(args.DA_model)
        else:
            Discriminator = DA1_discriminator
        if args.DA_model in ["DA2_discriminator", "DA3_discriminator"] and args.DA2_csize:
            discriminator = Discriminator(args.DA2_csize)
        else:
            discriminator = Discriminator()
        ssd_model = initSSD("ssd300",0.3,args.ssdpath)
        #target_encoder = ssd_model.extractor
        models = [discriminator, ssd_model]
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
                    t_train_iter = chainer.iterators.MultiprocessIterator(t_train, t_batchsize)
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

    else:
        if args.updater:
            Updater = eval(args.updater) #Updater1
        else:
            Updater = Updater1
        Discriminator = eval(args.adda_model)#ADDA_Discriminator2 #choose discriminator type
        discriminator = Discriminator()
        target_encoder = VGG16Extractor300()

        serializers.load_npz(args.initencoder, target_encoder)
        models = [discriminator, target_encoder]

    if args.discriminator_init:
        serializers.load_npz(args.discriminator_init,discriminator)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print("use gpu {}".format(args.gpu))
        for m in models:
            m.to_gpu()

    train_iter1 = chainer.iterators.MultiprocessIterator(source_dataset, args.batchsize)
    train_iter2 = chainer.iterators.MultiprocessIterator(target_dataset, args.batchsize)

    updater_args = {
        "iterator": {'main': train_iter1, 'target': train_iter2, },
        "device": args.gpu
    }

    if args.tgt_anno_data and args.s_t_ratio and args.s_t_ratio[0] != 0:
        updater_args['iterator']['tgt_annotation']= t_train_iter

    if args.mode in ["DA1_buf", "DA1_buf_multibatch"]:
        Updater = DA_updater1_buf
        if args.updater:
            Updater = eval(args.updater)
        if args.updater == "DA_updater1_buf_2":
            updater_args["bufmode"] = args.Alt_update_param[0]
            updater_args["batchmode"] = args.Alt_update_param[1]
            updater_args["cls_train_mode"] = args.Alt_update_param[2]
            if args.dis_step_init:
                updater_args["init_disstep"] = args.dis_step_init
            if args.tgt_step_init:
                updater_args["init_tgtstep"] = args.tgt_step_init
            if args.tgt_step_schedule:
                if len(args.tgt_step_schedule) % 2 != 0:
                    print("Warning: The number of argument of tgt step schedule is not an even number")
                tgt_step_schedule = []
                for i in range(int(len(args.tgt_step_schedule)/2)):
                    tgt_step_schedule.append([args.tgt_step_schedule[i*2],args.tgt_step_schedule[i*2+1]])
                updater_args["tgt_steps_schedule"] = tgt_step_schedule
            if args.fixed_source_dataset:
                updater_args["iterator"]['src_fmaps'] = chainer.iterators.MultiprocessIterator(COWC_fmap_set(args.fixed_source_dataset), args.batchsize)
        if args.mode == "DA1_buf_multibatch":
            Updater = DA_updater1_buf_multibatch
        if args.bufsize < int(args.batchsize/2):
            print("bufsize must not be smaller than batchsize/2")
            raise ValueError
        buffer = fmapBuffer(args.bufsize,mode=args.bufmode,discriminator=discriminator,gpu=args.gpu)
        updater_args["buffer"] = buffer
        if args.mode == "DA1_buf_multibatch" and args.multibatch_times:
            updater_args["n_multi_batch"] = args.multibatch_times
    if args.mode == "DA_fix_dis":
        Updater = DA_updater_enc_only

    # Set up optimizers
    opts["opt_dis"] = make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    if args.mode in ["DA1", "DA1_buf","DA1_buf_multibatch","DA_fix_dis"]:
        opts["opt_cls"] = make_optimizer(ssd_model, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    else:
        opts["opt_t_enc"] = make_optimizer(target_encoder, args.adam_alpha, args.adam_beta1, args.adam_beta2)


    updater_args["optimizer"] = opts
    updater_args["models"] = models

    # Set up updater and trainer
    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    # Set up logging
    # for m in models:
    #     trainer.extend(extensions.snapshot_object(
    #         m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.observe_lr(optimizer_name="opt_cls", observation_key='lr_cls'),
                   trigger=(args.display_interval, 'iteration'))
    if args.mode in ["DA1", "DA1_buf", "DA1_buf_multibatch"]:
        trainer.extend(extensions.observe_lr(optimizer_name="opt_cls",observation_key='lr_cls'),trigger=(args.display_interval, 'iteration'))
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

    # trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    # trainer.extend(extensions.ProgressBar(update_interval=10))

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