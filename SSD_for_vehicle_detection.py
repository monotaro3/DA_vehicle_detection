# coding: utf-8
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links.model.ssd.ssd_vgg16 import VGG16Extractor512, VGG16Extractor300,_imagenet_mean #_check_pretrained_model, _load_npz,
from chainercv.links.model.ssd import Multibox
import chainer
from chainer import Chain, initializers
import chainer.links as L
import chainer.functions as F
from chainercv import utils
from chainer import Variable, link_hooks
from chainercv.links.model.ssd import Normalize
# from chainercv.links.model.ssd.multibox_coder import _unravel_index


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

def ssd_predict_variable(ssd_model, imgs, raw = False):
    x = list()
    sizes = list()
    for img in imgs:
        _, H, W = img.shape
        img = ssd_model._prepare(img)
        x.append(ssd_model.xp.array(img))
        sizes.append((H, W))

    x = chainer.Variable(ssd_model.xp.stack(x))
    mb_locs, mb_confs = ssd_model(x)

    if raw:
        return mb_locs, mb_confs

    bboxes = list()
    labels = list()
    scores = list()
    for mb_loc, mb_conf, size in zip(mb_locs, mb_confs, sizes):
        # bbox, label, score = self.coder.decode(
        #     mb_loc, mb_conf, self.nms_thresh, self.score_thresh)
        # bbox = transforms.resize_bbox(
        #     bbox, (self.insize, self.insize), size)
        # bboxes.append(chainer.cuda.to_cpu(bbox))
        # labels.append(chainer.cuda.to_cpu(label))
        # scores.append(chainer.cuda.to_cpu(score))
        bbox, label, score = multibox_decode_variable(ssd_model.coder, mb_loc, mb_conf, ssd_model.nms_thresh, ssd_model.score_thresh)
        bbox = resize_bbox_variable(bbox, (ssd_model.insize, ssd_model.insize), size)
        bboxes.append(bbox)
        labels.append(label)
        scores.append(score)

    return bboxes, labels, scores


def multibox_decode_variable(coder, mb_loc, mb_conf, nms_thresh=0.45, score_thresh=0.6):
    # xp = self.xp
    #
    # # (center_y, center_x, height, width)
    # mb_bbox = self._default_bbox.copy()
    # mb_bbox[:, :2] += mb_loc[:, :2] * self._variance[0] \
    #                   * self._default_bbox[:, 2:]
    # mb_bbox[:, 2:] *= xp.exp(mb_loc[:, 2:] * self._variance[1])
    #
    # # (center_y, center_x, height, width) -> (y_min, x_min, height, width)
    # mb_bbox[:, :2] -= mb_bbox[:, 2:] / 2
    # # (center_y, center_x, height, width) -> (y_min, x_min, y_max, x_max)
    # mb_bbox[:, 2:] += mb_bbox[:, :2]
    xp = coder.xp
    mb_bbox_former = Variable(coder._default_bbox.copy().astype('f')[:,:2])
    mb_bbox_latter = Variable(coder._default_bbox.copy().astype('f')[:, 2:])
    mb_bbox_former += mb_loc[:, :2] * coder._variance[0] \
                          * coder._default_bbox[:, 2:]
    mb_bbox_latter *= F.exp(mb_loc[:, 2:] * coder._variance[1])
    mb_bbox_former -= mb_bbox_latter/2
    mb_bbox_latter += mb_bbox_former

    mb_bbox = F.hstack((mb_bbox_former,mb_bbox_latter))

    # mb_score = xp.exp(mb_conf)
    # mb_score /= mb_score.sum(axis=1, keepdims=True)
    mb_score = F.exp(mb_conf)
    mb_score /= F.hstack((F.sum(mb_score, axis=1, keepdims=True),)*2)

    bbox = list()
    label = list()
    score = list()
    # for l in range(mb_conf.shape[1] - 1):
    #     bbox_l = mb_bbox
    #     # the l-th class corresponds for the (l + 1)-th column.
    #     score_l = mb_score[:, l + 1]
    #
    #     mask = score_l >= score_thresh
    #     bbox_l = bbox_l[mask]
    #     score_l = score_l[mask]
    #
    #     if nms_thresh is not None:
    #         indices = utils.non_maximum_suppression(
    #             bbox_l, nms_thresh, score_l)
    #         bbox_l = bbox_l[indices]
    #         score_l = score_l[indices]
    #
    #     bbox.append(bbox_l)
    #     label.append(xp.array((l,) * len(bbox_l)))
    #     score.append(score_l)
    for l in range(mb_conf.shape[1] - 1):
        # score_l_ = mb_score[:, l + 1]
        # mask = score_l_.data >= score_thresh
        # if (mask == False).all():
        #     bbox_l = chainer.Variable(xp.array(()).astype('f'))
        #     score_l = chainer.Variable(xp.array(()).astype('f'))
        # else:
        #     bbox_l = []
        #     score_l = []
        #     for i in range(len(mask)):
        #         if mask[i]:
        #             bbox_l.append()
        bbox_l = mb_bbox
        # the l-th class corresponds for the (l + 1)-th column.
        score_l = mb_score[:, l + 1]

        mask = score_l.data >= score_thresh
        bbox_l = bbox_l[mask]
        score_l = score_l[mask]
        if nms_thresh is not None:
            indices = utils.non_maximum_suppression(
                        bbox_l.data, nms_thresh, score_l.data)
            bbox_l = bbox_l[indices]
            score_l = score_l[indices]
        bbox.append(bbox_l)
        label.append(Variable(xp.array((l,) * len(bbox_l)).astype(xp.int32)))
        score.append(score_l)
    # bbox = xp.vstack(bbox).astype(np.float32)
    # label = xp.hstack(label).astype(np.int32)
    # score = xp.hstack(score).astype(np.float32)
    bbox = F.vstack(bbox)
    label = F.hstack(label)
    score = F.hstack(score)

    return bbox, label, score

def multibox_encode_variable(coder, bbox, label, iou_thresh=0.5):
    xp = coder.xp

    if len(bbox) == 0:
        return (
            xp.zeros(coder._default_bbox.shape, dtype=np.float32),
            xp.zeros(coder._default_bbox.shape[0], dtype=np.int32))

    _default_bbox = coder._default_bbox.copy()
    # if xp != np:
    #     _default_bbox = chainer.cuda.to_cpu(_default_bbox)

    iou = utils.bbox_iou(
        xp.hstack((
            coder._default_bbox[:, :2] - coder._default_bbox[:, 2:] / 2,
            coder._default_bbox[:, :2] + coder._default_bbox[:, 2:] / 2)),
        bbox.data)

    index = xp.empty(len(coder._default_bbox), dtype=int)
    # -1 is for background
    index[:] = -1

    masked_iou = iou.copy()
    while True:
        # i, j = _unravel_index(masked_iou.argmax(), masked_iou.shape)
        i, j = xp.unravel_index(masked_iou.argmax(), masked_iou.shape)
        if masked_iou[i, j] <= 1e-6:
            break
        index[i] = j
        masked_iou[i, :] = 0
        masked_iou[:, j] = 0

    mask = xp.logical_and(index < 0, iou.max(axis=1) >= iou_thresh)
    if xp.count_nonzero(mask) > 0:
        index[mask] = iou[mask].argmax(axis=1)

    # mb_bbox = bbox[index].copy()
    # # (y_min, x_min, y_max, x_max) -> (y_min, x_min, height, width)
    # mb_bbox[:, 2:] -= mb_bbox[:, :2]
    # # (y_min, x_min, height, width) -> (center_y, center_x, height, width)
    # mb_bbox[:, :2] += mb_bbox[:, 2:] / 2
    #
    # mb_loc = xp.empty_like(mb_bbox)
    # mb_loc[:, :2] = (mb_bbox[:, :2] - self._default_bbox[:, :2]) / \
    #                 (self._variance[0] * self._default_bbox[:, 2:])
    # mb_loc[:, 2:] = xp.log(mb_bbox[:, 2:] / self._default_bbox[:, 2:]) / \
    #                 self._variance[1]
    #
    # # [0, n_fg_class - 1] -> [1, n_fg_class]
    # mb_label = label[index] + 1
    # # 0 is for background
    # mb_label[index < 0] = 0
    mb_bbox = bbox[index]
    mb_bbox_former = mb_bbox[:, :2]
    mb_bbox_latter = mb_bbox[:, 2:]
    mb_bbox_latter -= mb_bbox_former
    mb_bbox_former += mb_bbox_latter/2
    mb_loc_former = (mb_bbox_former - coder._default_bbox[:, :2]) / \
                    (coder._variance[0] * coder._default_bbox[:, 2:])
    mb_loc_latter = F.log(mb_bbox_latter / coder._default_bbox[:, 2:]) / \
                    coder._variance[1]
    mb_loc = F.hstack((mb_loc_former, mb_loc_latter))
    mb_label = label[index] + 1
    mb_label = F.where(index < 0, Variable(xp.zeros(mb_label.shape).astype(mb_label.data.dtype)),mb_label)

    return mb_loc, mb_label

def resize_bbox_variable(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :obj:`(y_min, x_min, y_max, x_max)`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    # bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox_0 = F.expand_dims(y_scale * bbox[:, 0],axis=1)
    bbox_2 = F.expand_dims(y_scale * bbox[:, 2],axis=1)
    bbox_1 = F.expand_dims(x_scale * bbox[:, 1],axis=1)
    bbox_3 = F.expand_dims(x_scale * bbox[:, 3],axis=1)
    return F.hstack((bbox_0,bbox_1,bbox_2,bbox_3))

class SSD512_vd(SSD512):
    def __init__(self, n_fg_class=None, pretrained_model=None,defaultbox_size=(35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6),mean = _imagenet_mean):
        # n_fg_class, path = _check_pretrained_model(
        #     n_fg_class, pretrained_model, self._models)
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        super(SSD512, self).__init__(
            extractor=VGG16Extractor512(),
            multibox=Multibox(
                # n_class=n_fg_class + 1,
                n_class=param['n_fg_class'] + 1,
                aspect_ratios=(
                    (2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 128, 256, 512),
            sizes=defaultbox_size,
            mean=mean)

        if path:
            # _load_npz(path, self)
            chainer.serializers.load_npz(path, self, strict=False)

class SSD300_vd(SSD300):
    def __init__(self, n_fg_class=None, pretrained_model=None,defaultbox_size=(30, 60, 111, 162, 213, 264, 315),mean = _imagenet_mean):
        # n_fg_class, path = _check_pretrained_model(
        #     n_fg_class, pretrained_model, self._models)
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        super(SSD300, self).__init__(
            extractor=VGG16Extractor300(),
            multibox=Multibox(
                # n_class=n_fg_class + 1,
                param['n_fg_class'] + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 100, 300),
            sizes=defaultbox_size,
            mean=mean)

        if path:
            # _load_npz(path, self)
            chainer.serializers.load_npz(path, self, strict=False)

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

class DA4_discriminator_sn(Chain):
    def __init__(self):
        #w = chainer.initializers.Normal(wscale)
        super(DA4_discriminator_sn, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(1024, 3, pad=1).add_hook(link_hooks.SpectralNormalization())
            self.conv2 = L.Convolution2D(512, 1).add_hook(link_hooks.SpectralNormalization())
            self.conv3 = L.Convolution2D(256, 1).add_hook(link_hooks.SpectralNormalization())
            self.conv4 = L.Convolution2D(1, 1).add_hook(link_hooks.SpectralNormalization())

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

class Recontructor(Chain):
    def __init__(self,upsample="deconv"):
        #w = chainer.initializers.Normal(wscale)
        self.upsample = upsample
        super(Recontructor, self).__init__()
        with self.init_scope():
            self.conv4_3 = L.Convolution2D(512,3,pad=1)
            self.conv4_2 = L.Convolution2D(512, 3, pad=1)
            self.conv4_1 = L.Convolution2D(512, 3, pad=1)
            self.conv3_3 = L.Convolution2D(256, 3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 3, pad=1)
            self.conv3_1 = L.Convolution2D(256, 3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 3, pad=1)
            self.conv2_1 = L.Convolution2D(128, 3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 3, pad=1)
            self.conv1_1 = L.Convolution2D(64, 3, pad=1)
            self.conv0 = L.Convolution2D(3, 3, pad=1)
            if self.upsample == "deconv":
                self.up_conv3 = L.Deconvolution2D(512,3,stride=2,pad=1)
                self.up_conv2 = L.Deconvolution2D(256, 3, stride=2, pad=1)
                self.up_conv1 = L.Deconvolution2D(128, 3, stride=2, pad=1)
            elif self.upsample == "unpool_conv":
                self.up_conv3 = L.Convolution2D(512,3,pad=1)
                self.up_conv2 = L.Convolution2D(256, 3, pad=1)
                self.up_conv1 = L.Convolution2D(128, 3, pad=1)

    def __call__(self, x):
        h = F.relu(self.conv4_3(x))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_1(h))
        if self.upsample == "deconv":
            h = F.relu(self.up_conv3(h))
        else:
            h = F.unpooling_2d(h,2)
        # h = F.pad(h, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant')
        if self.upsample == "unpool_conv":
            h = F.relu(self.up_conv3(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_1(h))
        if self.upsample == "deconv":
            h = F.relu(self.up_conv2(h))
        else:
            h = F.unpooling_2d(h,2)
        h = F.pad(h, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant')  #reflectpad can be an option
        if self.upsample == "unpool_conv":
            h = F.relu(self.up_conv2(h))
        h = F.relu(self.conv2_2(h))
        h = F.relu(self.conv2_1(h))
        if self.upsample == "deconv":
            h = F.relu(self.up_conv1(h))
        else:
            h = F.unpooling_2d(h,2)
        h = F.pad(h, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant')
        if self.upsample == "unpool_conv":
            h = F.relu(self.up_conv1(h))
        h = F.relu(self.conv1_2(h))
        h = F.relu(self.conv1_1(h))
        h = self.conv0(h)
        return h

class Generator_VGG16_simple(chainer.Chain):
    def __init__(self):
        super(Generator_VGG16_simple, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(64, 3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 3, pad=1)

            self.conv2_1 = L.Convolution2D(128, 3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 3, pad=1)

            self.conv3_1 = L.Convolution2D(256, 3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 3, pad=1)
            self.conv3_3 = L.Convolution2D(256, 3, pad=1)

            self.conv4_1 = L.Convolution2D(512, 3, pad=1)
            self.conv4_2 = L.Convolution2D(512, 3, pad=1)
            self.conv4_3 = L.Convolution2D(512, 3, pad=1)
            self.norm4 = Normalize(512, initial=initializers.Constant(20))

            # self.conv5_1 = L.DilatedConvolution2D(512, 3, pad=1)
            # self.conv5_2 = L.DilatedConvolution2D(512, 3, pad=1)
            # self.conv5_3 = L.DilatedConvolution2D(512, 3, pad=1)
            #
            # self.conv6 = L.DilatedConvolution2D(1024, 3, pad=6, dilate=6)
            # self.conv7 = L.Convolution2D(1024, 1)

    def forward(self, x):
        # ys = []

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        # ys.append(self.norm4(h))

        return self.norm4(h)

        # h = F.max_pooling_2d(h, 2)
        #
        # h = F.relu(self.conv5_1(h))
        # h = F.relu(self.conv5_2(h))
        # h = F.relu(self.conv5_3(h))
        # h = F.max_pooling_2d(h, 3, stride=1, pad=1)
        #
        # h = F.relu(self.conv6(h))
        # h = F.relu(self.conv7(h))
        # ys.append(h)
        #
        # return ys





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
        self.fmap_num = None
        # self.buffer_src_data = None
        # self.buffer_tgt_data = None

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

    def encode(self):
        if len(self.buffer_src) > 0:
            for i in range(len(self.buffer_src[0])):
                src_fmaps = [self.buffer_src[x][i] for x in range(len(self.buffer_src))]
                setattr(self,"buffer_src_data_"+str(i),np.array(src_fmaps))
                tgt_fmaps = [self.buffer_tgt[x][i] for x in range(len(self.buffer_tgt))]
                setattr(self,"buffer_tgt_data_" + str(i), np.array(tgt_fmaps))
            self.fmap_num = len(self.buffer_src[0])
        else:
            self.fmap_num = 0

    #     self.buffer_src_data = np.array(self.buffer_src)
    #     self.buffer_tgt_data = np.array(self.buffer_tgt)
    #
    def decode(self):
        if self.fmap_num > 0:
            buffer_src_data = []
            buffer_tgt_data = []
            # i = 0
            # src_temp = getattr(self,"buffer_src_data_"+str(i),None)
            # tgt_temp = getattr(self, "buffer_tgt_data_" + str(i),None)
            # while src_temp != None:
            for i in range(self.fmap_num):
                src_temp = getattr(self, "buffer_src_data_" + str(i), None)
                tgt_temp = getattr(self, "buffer_tgt_data_" + str(i),None)
                buffer_src_data.append(src_temp)
                buffer_tgt_data.append(tgt_temp)
            # i += 1
            # src_temp = getattr(self, "buffer_src_data_" + str(i), None)
            # tgt_temp = getattr(self, "buffer_tgt_data_" + str(i), None)
        # if len(buffer_src_data) > 0:
            self.buffer_src = [[buffer_src_data[x][y] for x in range(len(buffer_src_data))] for y in range(len(buffer_src_data[0]))]
            self.buffer_tgt = [[buffer_tgt_data[x][y] for x in range(len(buffer_tgt_data))] for y in
                               range(len(buffer_tgt_data[0]))]

    def serialize(self, serializer):
        if isinstance(serializer, chainer.serializer.Serializer):
            self.encode()
        self.fmap_num = serializer("fmap_num", self.fmap_num)
        # i = 0
        # src_temp = getattr(self, "buffer_src_data_" + str(i), None)
        # tgt_temp = getattr(self, "buffer_tgt_data_" + str(i), None)
        # while src_temp != None:
        # if isinstance(serializer, chainer.serializer.Deserializer):
        #     for i in range(self.fmap_num):
        #         setattr(self, "buffer_src_data_" + str(i), None)
        #         setattr(self, "buffer_tgt_data_" + str(i), None)
        for i in range(self.fmap_num):
            src_temp = getattr(self, "buffer_src_data_" + str(i), None)
            tgt_temp = getattr(self, "buffer_tgt_data_" + str(i), None)
            src_temp = serializer("buffer_src_data_" + str(i), src_temp)
            tgt_temp = serializer("buffer_tgt_data_" + str(i), tgt_temp)
            if isinstance(serializer, chainer.serializer.Deserializer):
                setattr(self, "buffer_src_data_" + str(i), src_temp)
                setattr(self, "buffer_tgt_data_" + str(i), tgt_temp)
            # buffer_src_data.append(src_temp)
            # buffer_tgt_data.append(tgt_temp)
            # i += 1
            # src_temp = getattr(self, "buffer_src_data_" + str(i), None)
            # tgt_temp = getattr(self, "buffer_tgt_data_" + str(i), None)
        if isinstance(serializer, chainer.serializer.Deserializer):
            self.decode()
        # self.rank_map_data = serializer('rank_map', self.rank_map_data)
        # self.rank_F1_data = serializer('rank_F1', self.rank_F1_data)
        # self.rank_mean_data = serializer('rank_mean', self.rank_mean_data)
        # if isinstance(serializer, chainer.serializer.Deserializer):
        #     self.rank_map = self.decode(self.rank_map_data)
        #     self.rank_F1 = self.decode(self.rank_F1_data)
        #     self.rank_mean = self.decode(self.rank_mean_data)
    #     buf = []
    #     for i in range(len(data)):
    #         buf.append(data[i])
    #         ranking_list.append([data[i][j] for j in range(data.shape[1])])
    #         ranking_list[i][0] = int(ranking_list[i][0]) #iter number must be int
    #     return ranking_list

class HistoricalBuffer():
    def __init__(self, buffer_size=50, image_size=256, image_channels=3, gpu = -1):
        self._buffer_size = buffer_size
        self._img_size = image_size
        self._img_ch = image_channels
        self._cnt = 0
        self.gpu = gpu
        import numpy
        import cupy
        xp = numpy if gpu < 0 else cupy
        self._buffer = xp.zeros((self._buffer_size, self._img_ch, self._img_size, self._img_size)).astype("f")

    def get_and_update(self, data, prob=0.5):
        if self._buffer_size == 0:
            return data
        xp = chainer.cuda.get_array_module(data)

        if len(data) == 1:
            if self._cnt < self._buffer_size:
                self._buffer[self._cnt,:] = chainer.cuda.to_cpu(data[0,:]) if self.gpu == -1 else data[0,:]
                self._cnt += 1
                return data
            else:
                if np.random.rand() > prob:
                    self._buffer[np.random.randint(self._cnt), :] = chainer.cuda.to_cpu(data[0,:]) if self.gpu == -1 else data[0,:]
                    return data
                else:
                    return xp.expand_dims(xp.asarray(self._buffer[np.random.randint(self._cnt),:]),axis=0)
        else:
            data = xp.copy(data)
            use_buf = len(data) // 2
            indices_rand = np.random.permutation(len(data))

            avail_buf = min(self._cnt, use_buf)
            if avail_buf > 0:
                indices_use_buf = np.random.choice(self._cnt,avail_buf,replace=False)
                data[indices_rand[-avail_buf:],:] = xp.asarray(self._buffer[indices_use_buf,:])
            room_buf = self._buffer_size - self._cnt
            n_replace_buf = min(self._cnt,len(data)-avail_buf-room_buf)
            if n_replace_buf > 0:
                indices_replace_buf = np.random.choice(self._cnt,n_replace_buf,replace=False)
                self._buffer[indices_replace_buf,:] =  chainer.cuda.to_cpu(data[indices_rand[-avail_buf-n_replace_buf:-avail_buf],:]) \
                    if self.gpu == -1 else data[indices_rand[-avail_buf-n_replace_buf:-avail_buf],:]
            if room_buf > 0:
                n_fill_buf = min(room_buf, len(data)-avail_buf)
                self._buffer[self._cnt:self._cnt+n_fill_buf,:] = chainer.cuda.to_cpu(data[indices_rand[0:n_fill_buf],:]) \
                    if self.gpu == -1 else data[indices_rand[0:n_fill_buf],:]
                self._cnt += n_fill_buf
            return data

    def serialize(self, serializer):
        self._cnt = serializer('cnt', self._cnt)
        self.gpu = serializer('gpu', self.gpu)
        self._buffer = serializer('buffer', self._buffer)



