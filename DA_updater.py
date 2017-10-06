import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


class Updater1(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.dis, self.t_enc = kwargs.pop('models')
        super(Updater1, self).__init__(*args, **kwargs)

    def update_core(self):
        t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_target = self.get_iterator('target').next()
        batchsize = len(batch_source)
        # x = []
        # for i in range(batchsize):
        #     x.append(np.asarray(batch[i]).astype("f"))
        # x_real = Variable(xp.asarray(x))
        source_fmap = []
        for i in range(6):
            fmap_ = []
            for j in range(batchsize):
                fmap_.append(batch_source[j][i])
            # if len(fmap_) == 1:
            #     fmap_[0] = fmap_[0][np.newaxis,:]
            #     fmap_ =
            # else:
            #     fmap_ = np.vstack(fmap_)
            fmap_ = xp.array(fmap_)
            source_fmap.append(Variable(fmap_))

        y_source = self.dis(source_fmap)

        t_fmap = self.t_enc(Variable(xp.array(batch_target)))
        y_target = self.dis(t_fmap)

        # z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        # x_fake = self.gen(z)
        # y_fake = self.dis(x_fake)

        # loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        # loss_dis += F.sum(F.softplus(y_fake)) / batchsize
        loss_dis = F.sum(F.softplus(-y_source)) / batchsize
        loss_dis += F.sum(F.softplus(y_target)) / batchsize

        loss_t_enc = F.sum(F.softplus(-y_target)) / batchsize

        self.t_enc.cleargrads()
        loss_t_enc.backward()
        t_enc_optimizer.update()
        for map in t_fmap:
            map.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_t_enc': loss_t_enc})
        chainer.reporter.report({'loss_dis': loss_dis})

class Updater2(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.dis, self.t_enc = kwargs.pop('models')
        super(Updater2, self).__init__(*args, **kwargs)

    def update_core(self):
        t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_target = self.get_iterator('target').next()
        batchsize = len(batch_source)
        # x = []
        # for i in range(batchsize):
        #     x.append(np.asarray(batch[i]).astype("f"))
        # x_real = Variable(xp.asarray(x))
        source_fmap = []
        for i in range(1):
            fmap_ = []
            for j in range(batchsize):
                fmap_.append(batch_source[j][i])
            # if len(fmap_) == 1:
            #     fmap_[0] = fmap_[0][np.newaxis,:]
            #     fmap_ =
            # else:
            #     fmap_ = np.vstack(fmap_)
            fmap_ = xp.array(fmap_)
            source_fmap.append(Variable(fmap_))

        y_source = self.dis(source_fmap)

        t_fmap = self.t_enc(Variable(xp.array(batch_target)))
        y_target = self.dis(t_fmap)

        # z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        # x_fake = self.gen(z)
        # y_fake = self.dis(x_fake)

        # loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        # loss_dis += F.sum(F.softplus(y_fake)) / batchsize
        loss_dis = F.sum(F.softplus(-y_source)) / batchsize
        loss_dis += F.sum(F.softplus(y_target)) / batchsize

        loss_t_enc = F.sum(F.softplus(-y_target)) / batchsize

        self.t_enc.cleargrads()
        loss_t_enc.backward()
        t_enc_optimizer.update()
        for map in t_fmap:
            map.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_t_enc': loss_t_enc})
        chainer.reporter.report({'loss_dis': loss_dis})