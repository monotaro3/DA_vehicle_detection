import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions
from chainer import serializers
from chainercv.links.model.ssd import VGG16Extractor300

from SSD_for_vehicle_detection import ADDA_Discriminator, ADDA_Discriminator2
from COWC_dataset_processed import COWC_fmap_set, Dataset_imgonly

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
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=10000, help='Interval of evaluation')
    parser.add_argument('--display_interval', type=int, default=10, help='Interval of displaying log to console')
    parser.add_argument('--n_dis', type=int, default=5, help='number of discriminator update per generator update')
    parser.add_argument('--gamma', type=float, default=0.5, help='hyperparameter gamma')
    parser.add_argument('--lam', type=float, default=10, help='gradient penalty')
    parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')
    parser.add_argument('--output_dim', type=int, default=256, help='output dimension of the discriminator (for cramer GAN)')
    parser.add_argument('--initencoder',  help='trained encoder which initializes target encoder')
    parser.add_argument('--adda_model', help='adda class name to be used')


    args = parser.parse_args()
    report_keys = ["loss_t_enc", "loss_dis"]

    # Set up dataset

    source_dataset = COWC_fmap_set("E:/work/vehicle_detection_dataset/cowc_300px_0.3_fmap")
    target_dataset = Dataset_imgonly("E:/work/vehicle_detection_dataset/Khartoum_adda")
    # train_iter1 = chainer.iterators.SerialIterator(source_dataset, args.batchsize)
    # train_iter2 = chainer.iterators.SerialIterator(target_dataset, args.batchsize)
    train_iter1 = chainer.iterators.MultiprocessIterator(source_dataset, args.batchsize)
    train_iter2 = chainer.iterators.MultiprocessIterator(target_dataset, args.batchsize)

    # Setup algorithm specific networks and updaters
    models = []
    opts = {}
    updater_args = {
        "iterator": {'main': train_iter1,'target': train_iter2,},
        "device": args.gpu
    }

    # if args.algorithm == "dcgan":
    #     from dcgan.updater import Updater
    #     if args.architecture=="dcgan":
    #         generator = common.net.DCGANGenerator()
    #         discriminator = common.net.DCGANDiscriminator()
    #     else:
    #         raise NotImplementedError()
    from DA_updater import Updater
    Discriminator = eval(args.adda_model)#ADDA_Discriminator2 #choose discriminator type
    discriminator = Discriminator()
    target_encoder = VGG16Extractor300()
    serializers.load_npz(args.initencoder, target_encoder)
    models = [discriminator, target_encoder]

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print("use gpu {}".format(args.gpu))
        for m in models:
            m.to_gpu()

    # Set up optimizers
    opts["opt_dis"] = make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    opts["opt_t_enc"] = make_optimizer(target_encoder, args.adam_alpha, args.adam_beta1, args.adam_beta2)

    updater_args["optimizer"] = opts
    updater_args["models"] = models

    # Set up updater and trainer
    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    # Set up logging
    for m in models:
        trainer.extend(extensions.snapshot_object(
            m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.snapshot(), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()