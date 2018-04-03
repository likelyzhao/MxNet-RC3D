"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

import numpy as np
import numpy.random as npr

from utils.image import get_image, tensor_vstack
from generate_anchor import generate_anchors,generate_anchors_twin
from bbox.bbox_transform import bbox_overlaps, bbox_transform
from twin.twin_transform import twin_transform,twin_overlaps

def get_rpn_testbatch(roidb, cfg):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    # assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs
    im_info = [np.array([roidb[i]['im_info']], dtype=np.float32) for i in range(len(roidb))]

    data = [{'data': im_array[i],
            'im_info': im_info[i]} for i in range(len(roidb))]
    label = {}

    return data, label, im_info

def get_rpn_batch(roidb, cfg):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    # gt boxes: (x1, y1, x2, y2, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_boxes': gt_boxes}

    return data, label

def prep_im_for_blob(im, pixel_means, target_size, crop_size, random_idx):
    import cv2
    """Mean subtract, resize and crop an frame for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_LINEAR)
    im -= pixel_means
    x = random_idx[1]
    y = random_idx[0]
    return im[x:x+crop_size, y:y+crop_size]

def _get_video_blob(roidb, scale_inds,cfg):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """
    # __C.TRAIN.LENGTH = (512,)
    import cv2
    DEBUG = False
    processed_videos = []
    video_scales = []
    print(scale_inds)
    for i,item in enumerate(roidb):
      video_length = cfg.network.MAXLENGTH[scale_inds[0]]
      video = np.zeros((video_length, cfg.network.CROP_SIZE,
                        cfg.network.CROP_SIZE, 3))
      if cfg.INPUT == 'video':
        j = 0
        random_idx = [np.random.randint(cfg.network.FRAME_SIZE[1]-cfg.network.CROP_SIZE),
                      np.random.randint(cfg.network.FRAME_SIZE[0]-cfg.network.CROP_SIZE)]
        for video_info in item['frames']:
          prefix = item['fg_name'] if video_info[0] else item['bg_name']
          for idx in xrange(video_info[1], video_info[2], video_info[3]):

            video_name = item['url'].split('v_')[-1].split('.mp4')[0]

            import os
            frame_name = os.path.join(prefix,video_name,'image_%s.jpg' %  str(idx + 1).zfill(5))
            print(frame_name)
#            frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))
            frame = cv2.imread(frame_name)

            frame = prep_im_for_blob(frame, cfg.network.PIXEL_MEANS, tuple(cfg.network.FRAME_SIZE[::-1]),
                                     cfg.network.CROP_SIZE, random_idx)

            if item['flipped']:
                frame = frame[:, ::-1, :]

            video[j] = frame
            j = j + 1

        while ( j < video_length):
          video[j] = frame
          j = j + 1

      else:
        j = 0
        random_idx = [np.random.randint(cfg.network.FRAME_SIZE[1]-cfg.network.CROP_SIZE),
                      np.random.randint(cfg.network.FRAME_SIZE[0]-cfg.network.CROP_SIZE)]
        for video_info in item['frames']:
          prefix = item['fg_name'] if video_info[0] else item['bg_name']
          for idx in xrange(video_info[1], video_info[2]):
            frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))
            frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.network.FRAME_SIZE[::-1]),
                                     cfg.network.CROP_SIZE, random_idx)

            if item['flipped']:
                frame = frame[:, ::-1, :]

            if DEBUG:
              cv2.imshow('frame', frame/255.0)
              cv2.waitKey(0)
              cv2.destroyAllWindows()

            video[j] = frame
            j = j + 1

        while ( j <= video_length):
          video[j] = frame
          j = j + 1
      processed_videos.append(video)

    # Create a blob to hold the input images
    # blob = video_list_to_blob(processed_videos)

    return processed_videos

def get_twin_rpn_batch(roidb, cfg):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    # imgs, roidb = get_image(roidb, cfg)
    num_videos = 1

    print(roidb)

    random_scale_inds = npr.randint(0, high=len(cfg.network.LENGTH),
                                    size=num_videos)
    assert (cfg.network.BATCH_SIZE % num_videos == 0), \
        'num_videos ({}) must divide BATCH_SIZE ({})'. \
            format(num_videos, cfg.network.BATCH_SIZE)

    im_array = _get_video_blob(roidb,random_scale_inds,cfg)
    print(type(im_array[0]))

    if roidb[0]['gt_classes'].size > 0:
        assert len(roidb) == 1, "Single batch only"
        # gt windows: (x1, x2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)
        gt_windows[:, 0:2] = roidb[0]['wins'][gt_inds, :]
        gt_windows[:, -1] = roidb[0]['gt_classes'][gt_inds]
    else: # not using RPN
        # Now, build the region of interest and label blobs
#        rois_blob = np.zeros((0, 3), dtype=np.float32)
#        labels_blob = np.zeros((0), dtype=np.float32)
#        bbox_targets_blob = np.zeros((0, 2 * num_classes), dtype=np.float32)
#        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        gt_windows = np.empty((0, 3), dtype=np.float32)

    data = {'data': im_array[0] }
    label = {'gt_windows': gt_windows}

    return data,label



    # im_array = imgs[0]

def assign_anchor_twin(feat_shape, gt_boxes, im_info, cfg, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    im_info = im_info[0]

#    anchor_scales = layer_params.get('scales', (2, 4, 5, 6, 8, 9, 10, 12, 14, 16))
#    self._anchors = generate_anchors(scales=np.array(anchor_scales))
#    self._num_anchors = self._anchors.shape[0]
#    self._feat_stride = layer_params['feat_stride']

    # map of shape (..., H, W)#
    length, height, width = im_info[0],im_info[1],im_info
    # GT boxes (x1, x2, label)
#    gt_boxes = bottom[1].data

    if DEBUG:
        print ''
        print 'length, height, width: ({}, {}, {})'.format(length, height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes



    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors_twin(base_size=feat_stride, scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    if DEBUG:
        print 'anchors:'
        print base_anchors
        print 'anchor shapes:'
        print np.hstack((base_anchors[:, 2::4] - base_anchors[:, 0::4],
                         base_anchors[:, 3::4] - base_anchors[:, 1::4]))
        print 'im_info', im_info
        print 'height', feat_height, 'width', feat_width
        print 'gt_boxes shape', gt_boxes.shape
        print 'gt_boxes', gt_boxes

    # 1. Generate proposals from twin deltas and shifted anchors
    shifts = np.arange(0, length) * feat_stride

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 2) shifted anchors
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = base_anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 1)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 2))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] < im_info[1] + allowed_border))[0]

    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    if DEBUG:
        print 'anchors shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0 and anchors.size >0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = twin_overlaps(np.ascontiguousarray(anchors, dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1


    twin_targets = np.zeros((len(inds_inside), 2), dtype=np.float32)

    # twin_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    if gt_boxes.size > 0 and anchors.size >0:
        twin_targets[:] = _compute_targets(anchors, twin_targets[argmax_overlaps, :4])

    twin_inside_weights = np.zeros((len(inds_inside), 2), dtype=np.float32)
    twin_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_TWIN_INSIDE_WEIGHTS)

    twin_outside_weights = np.zeros((len(inds_inside), 2), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 2)) * 1.0 / num_examples
        negative_weights = np.ones((1, 2)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    twin_outside_weights[labels == 1, :] = positive_weights
    twin_outside_weights[labels == 0, :] = negative_weights


    #bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    #bbox_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    if DEBUG:
        _sums = twin_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (twin_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(twin_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(twin_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(twin_outside_weights, total_anchors, inds_inside, fill=0)


    if DEBUG:
        print 'rpn: max max_overlaps', np.max(max_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    labels = labels.reshape((1, length, height, width, A)).transpose(0, 4, 1, 2, 3)
    labels = labels.reshape((1, 1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, length, height, width, A * 2)).transpose(0, 4, 3, 1, 2)
    bbox_inside_weights = bbox_inside_weights.reshape((1, length, height, width, A * 2)).transpose((0, 4, 3, 1, 2))
    bbox_outside_weights = bbox_outside_weights.reshape((1, length, height, width, A * 2)).transpose((0, 4, 3, 1, 2))

    label = {'label': labels,
             'bbox_target': bbox_targets,
             'bbox_inside_weight': bbox_inside_weights,
             'bbox_inside_weight': bbox_outside_weights}
    return label



def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 2
    assert gt_rois.shape[1] == 3

    return twin_transform(ex_rois, gt_rois[:, :2]).astype(np.float32, copy=False)


def assign_anchor(feat_shape, gt_boxes, im_info, cfg, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors(base_size=feat_stride, ratios=list(ratios), scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    if DEBUG:
        print 'anchors:'
        print base_anchors
        print 'anchor shapes:'
        print np.hstack((base_anchors[:, 2::4] - base_anchors[:, 0::4],
                         base_anchors[:, 3::4] - base_anchors[:, 1::4]))
        print 'im_info', im_info
        print 'height', feat_height, 'width', feat_width
        print 'gt_boxes shape', gt_boxes.shape
        print 'gt_boxes', gt_boxes

    # 1. generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] >= -allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + allowed_border) &
                           (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0 and anchors.size >0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0 and anchors.size >0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    if DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlaps', np.max(max_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
    bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))

    label = {'label': labels,
             'bbox_target': bbox_targets,
             'bbox_weight': bbox_weights}
    return label
