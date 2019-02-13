import chainer
from chainer import serializers
import copy
from chainercv.utils import apply_prediction_to_iterator

from COWC_dataset_processed import COWC_dataset_processed, vehicle_classes
from SSD_for_vehicle_detection import SSD300_vd, SSD512_vd
from SSD_for_vehicle_detection import defaultbox_size_300, defaultbox_size_512
from eval_detection_voc_custom import eval_detection_voc_custom

datadir = "e:/work/vehicle_detection_dataset/cowc_300px_0.3_daug_nmargin/"
ssd_path = "model/DA/m_thesis/tonly_x13_nmargin/model_iter_20000"#"model/300_0.3_daug_nmargin/model_iter_40000"
batchsize = 1
modelsize = "ssd300"
resolution = 0.3
gpu = 0

if modelsize == 'ssd300':
    model = SSD300_vd(
        n_fg_class=len(vehicle_classes),
        defaultbox_size=defaultbox_size_300[resolution])
else:
    model = SSD512_vd(
        n_fg_class=len(vehicle_classes),
        defaultbox_size=defaultbox_size_512[resolution])

serializers.load_npz(ssd_path, model)
if gpu >= 0: model.to_gpu()

test = COWC_dataset_processed(split="validation",datadir=datadir)
test_iter = chainer.iterators.SerialIterator(
    test, batchsize, repeat=False, shuffle=False)

if hasattr(test_iter, 'reset'):
    test_iter.reset()
    it = test_iter
else:
    it = copy.copy(test_iter)

imgs, pred_values, gt_values = apply_prediction_to_iterator(
            model.predict, it)
# delete unused iterator explicitly
del imgs

pred_bboxes, pred_labels, pred_scores = pred_values

if len(gt_values) == 3:
    gt_bboxes, gt_labels, gt_difficults = gt_values
elif len(gt_values) == 2:
    gt_bboxes, gt_labels = gt_values
    gt_difficults = None

print("predicting & evaluating...")
result, stats, matches, selec_list = eval_detection_voc_custom(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,use_07_metric=True,iou_thresh=0.4)

print(result)
print(stats)