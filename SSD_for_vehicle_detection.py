# coding: utf-8
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links.model.ssd.ssd_vgg16 import _check_pretrained_model, _load_npz, VGG16Extractor512, VGG16Extractor300,_imagenet_mean
from chainercv.links.model.ssd import Multibox
import chainer
from chainer import Chain

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

class Discriminator(Chain):
    def __init__(self, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():