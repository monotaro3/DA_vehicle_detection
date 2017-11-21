import argparse
from chainer import Variable, training
from chainer.training import extensions

from SSD_for_vehicle_detection import *
from COWC_dataset_processed import *

class dis_updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        #self.dis, self.cls = kwargs.pop('models')
        #self.buf = kwargs.pop('buffer')
        super(dis_updater, self).__init__(*args, **kwargs)
        #self.t_enc = self.cls.extractor
        self.discriminator = kwargs.pop('discriminator')

    def update_core(self):
        optimizer = self.get_optimizer("main")

        src_batch = self.get_iterator('main').next()
        tgt_batch = self.get_iterator('target').next()

        batchsize = len(src_batch)
        xp = self.discriminator.xp

        src_fmap = []
        tgt_fmap = []

        for i in range(len(src_batch[0])):
            src_fmap_ = []
            tgt_fmap_ = []
            for j in range(batchsize):
                src_fmap_.append(src_batch[j][i])
                tgt_fmap_.append(tgt_batch[j][i])
            # if len(fmap_) == 1:
            #     fmap_[0] = fmap_[0][np.newaxis,:]
            #     fmap_ =
            # else:
            #     fmap_ = np.vstack(fmap_)
            src_fmap_ = xp.array(src_fmap_)
            src_fmap.append(Variable(src_fmap_))
            tgt_fmap_ = xp.array(tgt_fmap_)
            tgt_fmap.append(Variable(tgt_fmap_))

        y_source = self.discriminator(src_fmap)
        y_target = self.discriminator(tgt_fmap)

        n_fmap_elements = y_target.shape[2] * y_target.shape[3]

        loss_dis_src = F.sum(F.softplus(-y_source)) / n_fmap_elements / batchsize
        loss_dis_tgt = F.sum(F.softplus(y_target)) / n_fmap_elements / batchsize
        loss_dis = loss_dis_src + loss_dis_tgt

        self.discriminator.cleargrads()
        loss_dis.backward()
        optimizer.update()

        chainer.reporter.report({'loss_dis': loss_dis})
        chainer.reporter.report({'loss_dis_src': loss_dis_src})
        chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt})

def main():
    parser = argparse.ArgumentParser(description='Train script')
    # parser.add_argument('--algorithm', '-a', type=str, default="dcgan", help='GAN algorithm')
    # parser.add_argument('--architecture', type=str, default="dcgan", help='Network architecture')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='pretrain_dis', help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10, help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10, help='Interval of displaying log to console')
    parser.add_argument('--discriminator', type=str, help='class name of discriminator')
    parser.add_argument('--src_fmap_data', type=str, help='src fmap data path')
    parser.add_argument('--tgt_fmap_data', type=str, help='tgt fmap data path')
    parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')


    args = parser.parse_args()

    report_keys = ["loss_dis",'loss_dis_src','loss_dis_tgt','lr']

    src_data = COWC_fmap_set(args.src_fmap_data)
    tgt_data = COWC_fmap_set(args.tgt_fmap_data)

    src_iter = chainer.iterators.MultiprocessIterator(src_data,args.batchsize)
    tgt_iter = chainer.iterators.MultiprocessIterator(tgt_data,args.batchsize)

    Discriminator_class = eval(args.discriminator)

    discriminator = Discriminator_class()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print("use gpu {}".format(args.gpu))
        discriminator.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=args.adam_alpha, beta1=args.adam_beta1, beta2=args.adam_beta2)
    optimizer.setup(discriminator)

    updater_args = {
        "iterator": {'main': src_iter, 'target': tgt_iter, },
        "device": args.gpu,
        "optimizer": optimizer,
        "discriminator" : discriminator
    }

    updater = dis_updater(**updater_args)
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    trainer.extend(extensions.observe_lr(),
                   trigger=(args.display_interval, 'iteration'))

    #trainer.extend(extensions.snapshot(), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        extensions.snapshot_object(discriminator, discriminator.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))

    trainer.run()

if __name__ == '__main__':
    main()

