# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
import cPickle


from core.rcnn import sample_rois
from twin.twin_transform import twin_overlaps,twin_transform
import numpy.random as npr

DEBUG = False

def _sample_rois(all_rois, gt_wins, fg_rois_per_image, rois_per_image, num_classes,cfg):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_wins)
    overlaps = twin_overlaps(
        np.ascontiguousarray(all_rois[:, 1:3], dtype=np.float),
        np.ascontiguousarray(gt_wins[:, :2], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_wins[gt_assignment, 2]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    twin_target_data = _compute_targets(
        rois[:, 1:3], gt_wins[gt_assignment[keep_inds], :2], labels)

    twin_targets, twin_inside_weights = \
        _get_twin_regression_labels(twin_target_data, num_classes)

    return labels, rois, twin_targets, twin_inside_weights


def _get_twin_regression_labels(twin_target_data, num_classes,cfg):
    """Bounding-box regression targets (twin_target_data) are stored in a
    compact form N x (class, tx, tl)
    This function expands those targets into the 4-of-2*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        twin_target (ndarray): N x 4K blob of regression targets
        twin_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = twin_target_data[:, 0]
    twin_targets = np.zeros((clss.size, 2 * num_classes), dtype=np.float32)
    twin_inside_weights = np.zeros(twin_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(2 * cls)
        end = start + 2
        twin_targets[ind, start:end] = twin_target_data[ind, 1:]
        twin_inside_weights[ind, start:end] = cfg.TRAIN.TWIN_INSIDE_WEIGHTS
    return twin_targets, twin_inside_weights

def _compute_targets(ex_rois, gt_rois, labels,cfg):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 2
    assert gt_rois.shape[1] == 2

    targets = twin_transform(ex_rois, gt_rois)
    if cfg.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.TWIN_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.TWIN_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_all_rois(all_rois, gt_wins, num_classes,cfg):
    """Generate all RoIs comprising foreground and background examples.
    """
    # overlaps: (rois x gt_wins)
    overlaps = twin_overlaps(
        np.ascontiguousarray(all_rois[:, 1:3], dtype=np.float),
        np.ascontiguousarray(gt_wins[:, :2], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_wins[gt_assignment, 2]

    labels = labels
    rois = all_rois

    twin_target_data = _compute_targets(
        rois[:, 1:3], gt_wins[gt_assignment, :2], labels, cfg)

    twin_targets, twin_inside_weights = \
        _get_twin_regression_labels(twin_target_data, num_classes,cfg)

    return labels, rois, twin_targets, twin_inside_weights


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._cfg = cfg
        self._fg_fraction = fg_fraction

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois == -1 or self._batch_rois % self._batch_images == 0, \
            'batchimages {} must devide batch_rois {}'.format(self._batch_images, self._batch_rois)
        all_rois = in_data[0].asnumpy()
        gt_wins = in_data[1].asnumpy()


        if self._batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_wins.shape[0]
            fg_rois_per_image = rois_per_image
        else:
            rois_per_image = self._batch_rois / self._batch_images
            fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)


        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_wins.shape[0], 1), dtype=gt_wins.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_wins[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        if self._sample == "All":
          labels, rois, twin_targets, twin_inside_weights = _sample_all_rois(
              all_rois, gt_wins, self._num_classes)
        else:
          # Sample rois with classification labels and bounding box regression
          # targets
          num_images = 1
          rois_per_image =  self._cfg.TRAIN.BATCH_SIZE / num_images
          fg_rois_per_image = int(round(self._cfg.TRAIN.FG_FRACTION * rois_per_image))
          labels, rois, twin_targets, twin_inside_weights = _sample_rois(
              all_rois, gt_wins, fg_rois_per_image,
              rois_per_image, self._num_classes)

#        rois, labels, bbox_targets, bbox_weights = \
#            _sample_all_rois(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, self._cfg, gt_boxes=gt_boxes,cfg=self._cfg)

        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        for ind, val in enumerate([rois, labels, twin_targets, twin_inside_weights]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('proposal_target_twin')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction='0.25'):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._cfg = cPickle.loads(cfg)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        rois = rpn_rois_shape[0] + gt_boxes_shape[0] if self._batch_rois == -1 else self._batch_rois

        output_rois_shape = (rois, 5)
        label_shape = (rois, )
        bbox_target_shape = (rois, self._num_classes * 4)
        bbox_weight_shape = (rois, self._num_classes * 4)

        return [rpn_rois_shape, gt_boxes_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._cfg, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
