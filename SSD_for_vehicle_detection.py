# coding: utf-8
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links.model.ssd.ssd_vgg16 import _check_pretrained_model, _load_npz, VGG16Extractor512, VGG16Extractor300,_imagenet_mean
from chainercv.links.model.ssd import Multibox
import chainer
from chainer import Chain, initializers
import chainer.links as L
import chainer.functions as F


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

class SSD512_vd(SSD512):
    def __init__(self, n_fg_class=None, pretrained_model=None,defaultbox_size=(35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6),mean = _imagenet_mean):
        n_fg_class, path = _check_pretrained_model(
            n_fg_class, pretrained_model, self._models)

        super(SSD512, self).__init__(
            extractor=VGG16Extractor512(),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=(
                    (2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 128, 256, 512),
            sizes=defaultbox_size,
            mean=mean)

        if path:
            _load_npz(path, self)

class SSD300_vd(SSD300):
    def __init__(self, n_fg_class=None, pretrained_model=None,defaultbox_size=(30, 60, 111, 162, 213, 264, 315),mean = _imagenet_mean):
        n_fg_class, path = _check_pretrained_model(
            n_fg_class, pretrained_model, self._models)

        super(SSD300, self).__init__(
            extractor=VGG16Extractor300(),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 100, 300),
            sizes=defaultbox_size,
            mean=mean)

        if path:
            _load_npz(path, self)



class ADDA_Discriminator(Chain):
    def __init__(self, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(ADDA_Discriminator, self).__init__()
        init = {
            'initialW': initializers.LeCunUniform(),
            'initial_bias': initializers.Zero(),
        }
        with self.init_scope():
            self.conv5_1 = L.DilatedConvolution2D(512, 3, pad=1)
            self.conv5_2 = L.DilatedConvolution2D(512, 3, pad=1)
            self.conv5_3 = L.DilatedConvolution2D(512, 3, pad=1)

            self.conv6 = L.DilatedConvolution2D(1024, 3, pad=6, dilate=6)
            self.conv7 = L.Convolution2D(1024, 1)
            self.conv8_1 = L.Convolution2D(256, 1, **init)
            self.conv8_2 = L.Convolution2D(512, 3, stride=2, pad=1, **init)

            self.conv9_1 = L.Convolution2D(128, 1, **init)
            self.conv9_2 = L.Convolution2D(256, 3, stride=2, pad=1, **init)

            self.conv10_1 = L.Convolution2D(128, 1, **init)
            self.conv10_2 = L.Convolution2D(256, 3, **init)

            self.conv11_1 = L.Convolution2D(128, 1, **init)
            self.conv11_2 = L.Convolution2D(256, 3, **init)

            self.l = L.Linear(256, 1, initialW=w)

    def __call__(self, x):
        h = F.max_pooling_2d(x[0], 2)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h += x[1]
        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        h += x[2]
        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        h += x[3]
        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        h += x[4]
        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        h += x[5]
        return self.l(h)

class ADDA_Discriminator2(Chain):
    def __init__(self, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(ADDA_Discriminator2, self).__init__()
        init = {
            'initialW': initializers.LeCunUniform(),
            'initial_bias': initializers.Zero(),
        }
        with self.init_scope():
            self.conv5_1 = L.DilatedConvolution2D(512, 3, pad=1)
            self.conv5_2 = L.DilatedConvolution2D(512, 3, pad=1)
            self.conv5_3 = L.DilatedConvolution2D(512, 3, pad=1)
            self.scale1 = L.Scale(W_shape=(1))

            self.conv6 = L.DilatedConvolution2D(1024, 3, pad=6, dilate=6)
            self.conv7 = L.Convolution2D(1024, 1)
            self.conv8_1 = L.Convolution2D(256, 1, **init)
            self.conv8_2 = L.Convolution2D(512, 3, stride=2, pad=1, **init)
            self.scale2 = L.Scale(W_shape=(1))

            self.conv9_1 = L.Convolution2D(128, 1, **init)
            self.conv9_2 = L.Convolution2D(256, 3, stride=2, pad=1, **init)
            self.scale3 = L.Scale(W_shape=(1))

            self.conv10_1 = L.Convolution2D(128, 1, **init)
            self.conv10_2 = L.Convolution2D(256, 3, **init)
            self.scale4 = L.Scale(W_shape=(1))

            self.conv11_1 = L.Convolution2D(128, 1, **init)
            self.conv11_2 = L.Convolution2D(256, 3, **init)
            self.scale5 = L.Scale(W_shape=(1))

            self.l = L.Linear(256, 1, initialW=w)

    def __call__(self, x):
        h = F.max_pooling_2d(x[0], 2)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)
        h = F.relu(self.conv6(h))
        h = self.scale1(F.relu(self.conv7(h)))
        h += x[1]
        h = F.relu(self.conv8_1(h))
        h = self.scale2(F.relu(self.conv8_2(h)))
        h += x[2]
        h = F.relu(self.conv9_1(h))
        h = self.scale3(F.relu(self.conv9_2(h)))
        h += x[3]
        h = F.relu(self.conv10_1(h))
        h = self.scale4(F.relu(self.conv10_2(h)))
        h += x[4]
        h = F.relu(self.conv11_1(h))
        h = self.scale5(F.relu(self.conv11_2(h)))
        h += x[5]
        return self.l(h)

class ADDA_Discriminator3(Chain):
    def __init__(self, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(ADDA_Discriminator3, self).__init__()
        init = {
            'initialW': initializers.LeCunUniform(),
            'initial_bias': initializers.Zero(),
        }
        with self.init_scope():
            self.conv1 = L.Convolution2D(512, 8)
            self.conv2 = L.Convolution2D(512, 8)
            self.conv3 = L.Convolution2D(512, 8)
            self.conv4 = L.Convolution2D(512, 8)
            self.conv5 = L.Convolution2D(512, 8)
            self.conv6 = L.Convolution2D(256, 3)
            self.l = L.Linear(256, 1, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x[0]))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.leaky_relu(self.conv5(h))
        h = F.leaky_relu(self.conv6(h))
        h = F.leaky_relu(self.l(h))
        return h

class ADDA_Discriminator4(Chain):
    def __init__(self, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(ADDA_Discriminator4, self).__init__()
        init = {
            'initialW': initializers.LeCunUniform(),
            'initial_bias': initializers.Zero(),
        }
        with self.init_scope():
            self.conv1 = L.Convolution2D(512, 7)
            self.conv2 = L.Convolution2D(512, 7)
            self.conv3 = L.Convolution2D(256, 5)
            self.l = L.Linear(256, 1, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x[0]))
        h = F.max_pooling_2d(h,2)
        h = F.leaky_relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.leaky_relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.leaky_relu(self.l(h))
        return h

class DA1_discriminator(Chain):
    def __init__(self):
        #w = chainer.initializers.Normal(wscale)
        super(DA1_discriminator, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 3, pad=1)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x[0]))
        return h


class DA2_discriminator(Chain):
    def __init__(self, c_size = 256):
        #w = chainer.initializers.Normal(wscale)
        super(DA2_discriminator, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(c_size, 3, pad=1)
            self.conv2 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x[0]))
        h = F.leaky_relu(self.conv2(h))
        return h

class DA2_discriminator_bn(Chain):
    def __init__(self, c_size = 256):
        #w = chainer.initializers.Normal(wscale)
        super(DA2_discriminator_bn, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(c_size, 3, pad=1)
            self.bn1 = L.BatchNormalization(c_size)
            self.conv2 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x[0])))
        h = F.leaky_relu(self.conv2(h))
        return h

class DA3_discriminator(Chain):
    def __init__(self, c_size = 256):
        #w = chainer.initializers.Normal(wscale)
        super(DA3_discriminator, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(c_size, 3, pad=1)
            self.conv2 = L.Convolution2D(c_size, 1)
            self.conv3 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x[0]))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        return h

class DA3_discriminator_bn(Chain):
    def __init__(self, c_size = 256):
        #w = chainer.initializers.Normal(wscale)
        super(DA3_discriminator_bn, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(c_size, 3, pad=1)
            self.bn1 = L.BatchNormalization(c_size)
            self.conv2 = L.Convolution2D(c_size, 1)
            self.bn2 = L.BatchNormalization(c_size)
            self.conv3 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x[0])))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.conv3(h))
        return h

class DA4_discriminator(Chain):
    def __init__(self):
        #w = chainer.initializers.Normal(wscale)
        super(DA4_discriminator, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(1024, 3, pad=1)
            self.conv2 = L.Convolution2D(512, 1)
            self.conv3 = L.Convolution2D(256, 1)
            self.conv4 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x[0]))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        return h

class DA4_discriminator_bn(Chain):
    def __init__(self):
        #w = chainer.initializers.Normal(wscale)
        super(DA4_discriminator_bn, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(1024, 3, pad=1)
            self.bn1 = L.BatchNormalization(1024)
            self.conv2 = L.Convolution2D(512, 1)
            self.bn2 = L.BatchNormalization(512)
            self.conv3 = L.Convolution2D(256, 1)
            self.bn3 = L.BatchNormalization(256)
            self.conv4 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x[0])))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.leaky_relu(self.conv4(h))
        return h

import random
import numpy as np
import math

class fmapBuffer(object):
    def __init__(self,bufsize,mode =0,discriminator=None,batchsize = 32,gpu = -1):  #mode 0:align src and tgt, 1:not align, 2:sort by loss value
        self.bufsize = bufsize
        self.buffer_src = []
        self.buffer_tgt = []
        self.mode = mode
        self.discriminator = discriminator
        self.loss_src = None
        self.loss_tgt = None
        self.batchsize = batchsize
        self.gpu = gpu

    def get_examples(self, n_samples):
        if self.buffer_src == []:
            n_return, src_samples, tgt_samples = 0, None, None
        elif n_samples >= len(self.buffer_src):
            n_return = len(self.buffer_src)
            n_fmap = len(self.buffer_src[0])
            src_samples = []
            tgt_samples = []
            for i in range(n_fmap):
                src_samples.append(np.stack([x[i] for x in self.buffer_src]))
                tgt_samples.append(np.stack([x[i] for x in self.buffer_tgt]))
        else:
            n_return = n_samples
            if self.mode == 2:
                indices_src = [x for x in range(n_samples)]
                indices_tgt = indices_src
            else:
                indices_src = random.sample(range(len(self.buffer_src)), n_samples)
                if self.mode == 0:
                    indices_tgt = indices_src
                else:
                    indices_tgt = random.sample(range(len(self.buffer_tgt)), n_samples)
            n_fmap = len(self.buffer_src[0])
            src_samples = []
            tgt_samples = []
            for i in range(n_fmap):
                src_samples.append(np.stack([self.buffer_src[x][i] for x in indices_src]))
                tgt_samples.append(np.stack([self.buffer_tgt[x][i] for x in indices_tgt]))
        return n_return, src_samples, tgt_samples

    def set_examples(self, src_samples, tgt_samples):
        n_samples = src_samples[0].shape[0]
        if self.bufsize < n_samples:
            print(self.__class__.__name__+"- set_example(): n_examples must not be larger than bufsize ")
            raise ValueError
        n_fmap = len(src_samples)
        n_fmap_elements = src_samples[0].shape[2] * src_samples[0].shape[3]
        if self.mode == 2:
            for i in range(n_samples):
                self.buffer_src.append([src_samples[x][i] for x in range(n_fmap)])
                self.buffer_tgt.append([tgt_samples[x][i] for x in range(n_fmap)])
            del src_samples
            del tgt_samples
            transfer_array = lambda x: chainer.cuda.to_gpu(x, device=self.gpu) if self.gpu >= 0 else lambda x: x
            for i in range(int(math.ceil(len(self.buffer_src)/self.batchsize))):
                src_examples_ = []
                tgt_examples_ = []
                for j in range(n_fmap):
                    src_examples_.append(transfer_array(np.stack([x[j] for x in self.buffer_src[i*self.batchsize:(i+1)*self.batchsize]])))
                    tgt_examples_.append(transfer_array(np.stack([x[j] for x in self.buffer_tgt[i * self.batchsize:(i + 1) * self.batchsize]])))
                with chainer.no_backprop_mode():
                    src_loss_ = chainer.cuda.to_cpu(F.sum(F.softplus(-self.discriminator(src_examples_)),axis=(1,2,3)).data) / n_fmap_elements
                    tgt_loss_ = chainer.cuda.to_cpu(
                        F.sum(F.softplus(self.discriminator(tgt_examples_)), axis=(1, 2, 3)).data) / n_fmap_elements
                if i == 0:
                    self.loss_src = src_loss_
                    self.loss_tgt = tgt_loss_
                else:
                    self.loss_src = np.hstack((self.loss_src, src_loss_))
                    self.loss_tgt = np.hstack((self.loss_tgt, tgt_loss_))
            # self.buffer_src = sorted(self.buffer_src, key=lambda x: self.loss_src[self.buffer_src.index(x)],reverse=True)
            # self.buffer_tgt = sorted(self.buffer_tgt, key=lambda x: self.loss_tgt[self.buffer_tgt.index(x)],reverse=True)
            index_sorted_src = np.argsort(self.loss_src)[::-1]
            index_sorted_tgt = np.argsort(self.loss_tgt)[::-1]
            self.buffer_src = [self.buffer_src[x] for x in index_sorted_src]
            self.buffer_tgt = [self.buffer_tgt[x] for x in index_sorted_tgt]
            self.loss_src = np.sort(self.loss_src)[::-1]
            self.loss_tgt = np.sort(self.loss_tgt)[::-1]
            self.buffer_src = self.buffer_src[:self.bufsize]
            self.buffer_tgt = self.buffer_tgt[:self.bufsize]
            self.loss_src = self.loss_src[:self.bufsize]
            self.loss_tgt = self.loss_tgt[:self.bufsize]
        else:
            n_room = self.bufsize - len(self.buffer_src)
            if n_room >= n_samples:
                for i in range(n_samples):
                    self.buffer_src.append([src_samples[x][i] for x in range(n_fmap)])
                    self.buffer_tgt.append([tgt_samples[x][i] for x in range(n_fmap)])
            else:
                indices_buf_src = random.sample(range(len(self.buffer_src)), n_samples - n_room)
                if self.mode == 0:
                    indices_tgt_src = indices_buf_src
                else:
                    indices_tgt_src = random.sample(range(len(self.buffer_tgt)), n_samples - n_room)
                indices_samples = range(n_room,n_samples)
                for i,j,k in zip(indices_buf_src,indices_tgt_src, indices_samples):
                    self.buffer_src[i] = [src_samples[x][k] for x in range(n_fmap)]
                    self.buffer_tgt[j] = [tgt_samples[x][k] for x in range(n_fmap)]
                for i in range(n_room):
                    self.buffer_src.append([src_samples[x][i] for x in range(n_fmap)])
                    self.buffer_tgt.append([tgt_samples[x][i] for x in range(n_fmap)])




