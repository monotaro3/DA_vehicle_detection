# coding: utf-8
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links.model.ssd.ssd_vgg16 import _check_pretrained_model, _load_npz, VGG16Extractor512, VGG16Extractor300,_imagenet_mean
from chainercv.links.model.ssd import Multibox
import chainer
from chainer import Chain, initializers
import chainer.links as L
import chainer.functions as F

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


