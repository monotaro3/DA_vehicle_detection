# coding: utf-8
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links.model.ssd.ssd_vgg16 import _check_pretrained_model, _load_npz, VGG16Extractor512, _imagenet_mean
from chainercv.links.model.ssd import Multibox

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