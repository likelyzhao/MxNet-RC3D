"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path. As to training, it is necessary that ground
truth and proposals are mixed together for training.
roidb
basic format [image_index]
['image', 'height', 'width', 'flipped',
'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import os
import cPickle
import numpy as np
from PIL import Image
from bbox.bbox_transform import bbox_overlaps
from multiprocessing import Pool, cpu_count
import json

def get_flipped_entry_outclass_wrapper(IMDB_instance, seg_rec):
    return IMDB_instance.get_flipped_entry(seg_rec)

path = './preprocess/activityNet/frames/'

def generate_roi(rois, start, end, stride):
    split = 'train'
    video = '1'
    tmp = {}
    tmp['wins'] = ( rois[:,:2] - start ) / stride
    tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0]
    tmp['gt_classes'] = rois[:,2]
    tmp['max_classes'] = rois[:,2]
    tmp['max_overlaps'] = np.ones(len(rois))
    tmp['flipped'] = False
    tmp['frames'] = np.array([[0, start, end, stride]])
    tmp['bg_name'] = path + split + '/' + video
    tmp['fg_name'] = path + split + '/' + video
    if not os.path.isfile('../../' + tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'):
        print('../../' + tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg')
    raise
    return tmp

class VIDEODB(object):
    def __init__(self, name, image_set, root_path, dataset_path, result_path=None):
        """
        basic information about an image database
        :param name: name of image database will be used for any output
        :param root_path: root path store cache and proposal data
        :param dataset_path: dataset path store images and image lists
        """
        self.name = name + '_' + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self._result_path = result_path

        # abstract attributes
        self.classes = []
        self.num_classes = 0
        self.image_set_index = []
        self.num_images = 0

        self.config = {}

    def image_path_from_index(self, index):
        raise NotImplementedError

    def gt_roidb(self,json_path):

        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.create_roidb_from_json_list(json_path)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def evaluate_detections(self, detections):
        raise NotImplementedError

    def evaluate_segmentations(self, segmentations):
        raise NotImplementedError

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.root_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    @property
    def result_path(self):
        if self._result_path and os.path.exists(self._result_path):
            return self._result_path
        else:
            return self.cache_path

    def image_path_at(self, index):
        """
        access image at index in image database
        :param index: image index in image database
        :return: image path
        """
        return self.image_path_from_index(self.image_set_index[index])

    def load_rpn_data(self, full=False):
        if full:
            rpn_file = os.path.join(self.result_path, 'rpn_data', self.name + '_full_rpn.pkl')
        else:
            rpn_file = os.path.join(self.result_path, 'rpn_data', self.name + '_rpn.pkl')
        print 'loading {}'.format(rpn_file)
        assert os.path.exists(rpn_file), 'rpn data not found at {}'.format(rpn_file)
        with open(rpn_file, 'rb') as f:
            box_list = cPickle.load(f)
        return box_list

    def load_rpn_roidb(self, gt_roidb):
        """
        turn rpn detection boxes into roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        box_list = self.load_rpn_data()
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def rpn_roidb(self, gt_roidb, append_gt=False):
        """
        get rpn roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of rpn
        """
        if append_gt:
            print 'appending ground truth annotations'
            rpn_roidb = self.load_rpn_roidb(gt_roidb)
            roidb = VIDEODB.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self.load_rpn_roidb(gt_roidb)
        return roidb

    def generate_classes(self, dict_list):
        class_list = []
        for dict in dict_list:
            for item in dict['clips']['data']:
                if item not in class_list:
                    class_list.append(item['label'])

        class_list = list(set(class_list))
        classes = {'Background': 0}
        for i, cls in enumerate(class_list):
            classes[cls] = i + 1
        return classes


    def create_roidb_from_json_list(self, json_list_path):
        """
        given ground truth, prepare roidb
        :param json_list_path: [image_index] ndarray of [box_index][x1, x2, y1, y2]
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        # assert len(box_list) == self.num_images, 'number of boxes matrix must match number of images'

        import json
        dicts = []
        with open(json_list_path) as f:
            dicts = dicts.append(json.loads(f.readline()))

        self.classes = VIDEODB.generate_classes(dicts)
 #       segment = VIDEODB.generate_segment()

        FPS = 25
        LENGTH = 768
        min_length = 3
        overlap_thresh = 0.7
        STEP = LENGTH / 4
        WINS = [LENGTH * 8]
        USE_FLIPPED = True

        duration = []
        roidb = []
        for dict in dicts:
            length = dict['metadata']['duration']
            segment =[]

            for anno in dict['clips'][0]['data']:
                start_time = anno['segment'][0]
                end_time = anno['segment'][1]
                label = self.classes[anno['label']]
                segment.append([start_time, end_time, label])

            segment.sort(key=lambda x: x[0])

            db = np.array(segment)
            if len(db) == 0:
                continue
            db[:, :2] = db[:, :2] * FPS

            for win in WINS:
                stride = win / LENGTH
                step = stride * STEP
                # Forward Direction
                for start in xrange(0, max(1, length - win + 1), step):
                    end = min(start + win, length)
                    assert end <= length
                    rois = db[np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))]

                    # Remove duration less than min_length
                    if len(rois) > 0:
                        duration = rois[:, 1] - rois[:, 0]
                        rois = rois[duration >= min_length]

                    # Remove overlap less than overlap_thresh
                    if len(rois) > 0:
                        time_in_wins = (np.minimum(end, rois[:, 1]) - np.maximum(start, rois[:, 0])) * 1.0
                        overlap = time_in_wins / (rois[:, 1] - rois[:, 0])
                        assert min(overlap) >= 0
                        assert max(overlap) <= 1
                        rois = rois[overlap >= overlap_thresh]

                    # Append data
                    if len(rois) > 0:
                        rois[:, 0] = np.maximum(start, rois[:, 0])
                        rois[:, 1] = np.minimum(end, rois[:, 1])
                        tmp = generate_roi(rois, start, end, stride)
                        roidb.append(tmp)
                        if USE_FLIPPED:
                            flipped_tmp = copy.deepcopy(tmp)
                            flipped_tmp['flipped'] = True
                            roidb.append(flipped_tmp)

                # Backward Direction
                for end in xrange(length, win - 1, - step):
                    start = end - win
                    assert start >= 0
                    rois = db[np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))]

                    # Remove duration less than min_length
                    if len(rois) > 0:
                        duration = rois[:, 1] - rois[:, 0]
                        rois = rois[duration > min_length]

                    # Remove overlap less than overlap_thresh
                    if len(rois) > 0:
                        time_in_wins = (np.minimum(end, rois[:, 1]) - np.maximum(start, rois[:, 0])) * 1.0
                        overlap = time_in_wins / (rois[:, 1] - rois[:, 0])
                        assert min(overlap) >= 0
                        assert max(overlap) <= 1
                        rois = rois[overlap > overlap_thresh]

                    # Append data
                    if len(rois) > 0:
                        rois[:, 0] = np.maximum(start, rois[:, 0])
                        rois[:, 1] = np.minimum(end, rois[:, 1])
                        tmp = generate_roi(rois, start, end, stride)
                        roidb.append(tmp)
                        if USE_FLIPPED:
                            import copy
                            flipped_tmp = copy.deepcopy(tmp)
                            flipped_tmp['flipped'] = True
                            roidb.append(flipped_tmp)

        return roidb


    def creat_label(self,des):
        pass

    def get_flipped_entry(self, seg_rec):
        return {'image': self.flip_and_save(seg_rec['image']),
                'seg_cls_path': self.flip_and_save(seg_rec['seg_cls_path']),
                'height': seg_rec['height'],
                'width': seg_rec['width'],
                'flipped': True}

    def append_flipped_images_for_segmentation(self, segdb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param segdb: [image_index]['seg_cls_path', 'flipped']
        :return: segdb: [image_index]['seg_cls_path', 'flipped']
        """
        print 'append flipped images to segdb'
        assert self.num_images == len(segdb)
        pool = Pool(processes=1)
        pool_result = []
        for i in range(self.num_images):
            seg_rec = segdb[i]
            pool_result.append(pool.apply_async(get_flipped_entry_outclass_wrapper, args=(self, seg_rec, )))
            #self.get_flipped_entry(seg_rec, segdb_flip, i)
        pool.close()
        pool.join()
        segdb_flip = [res_instance.get() for res_instance in pool_result]
        segdb += segdb_flip
        self.image_set_index *= 2
        return segdb

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print 'append flipped images to roidb'
        assert self.num_images == len(roidb)
        for i in range(self.num_images):
            roi_rec = roidb[i]
            boxes = roi_rec['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = roi_rec['width'] - oldx2 - 1
            boxes[:, 2] = roi_rec['width'] - oldx1 - 1
            print(boxes)
            print(roi_rec['image'])
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'image': roi_rec['image'],
                     'height': roi_rec['height'],
                     'width': roi_rec['width'],
                     'boxes': boxes,
                     'gt_classes': roidb[i]['gt_classes'],
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'flipped': True}

            # if roidb has mask
            if 'cache_seg_inst' in roi_rec:
                [filename, extension] = os.path.splitext(roi_rec['cache_seg_inst'])
                entry['cache_seg_inst'] = os.path.join(filename + '_flip' + extension)

            roidb.append(entry)

        self.image_set_index *= 2
        return roidb

    def flip_and_save(self, image_path):
        """
        flip the image by the path and save the flipped image with suffix 'flip'
        :param path: the path of specific image
        :return: the path of saved image
        """
        [image_name, image_ext] = os.path.splitext(os.path.basename(image_path))
        image_dir = os.path.dirname(image_path)
        saved_image_path = os.path.join(image_dir, image_name + '_flip' + image_ext)
        try:
            flipped_image = Image.open(saved_image_path)
        except:
            flipped_image = Image.open(image_path)
            flipped_image = flipped_image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image.save(saved_image_path, 'png')
        return saved_image_path

    def evaluate_recall(self, roidb, candidate_boxes=None, thresholds=None):
        """
        evaluate detection proposal recall metrics
        record max overlap value for each gt box; return vector of overlap values
        :param roidb: used to evaluate
        :param candidate_boxes: if not given, use roidb's non-gt boxes
        :param thresholds: array-like recall threshold
        :return: None
        ar: average recall, recalls: vector recalls at each IoU overlap threshold
        thresholds: vector of IoU overlap threshold, gt_overlaps: vector of all ground-truth overlaps
        """
        all_log_info = ''
        area_names = ['all', '0-25', '25-50', '50-100',
                      '100-200', '200-300', '300-inf']
        area_ranges = [[0**2, 1e5**2], [0**2, 25**2], [25**2, 50**2], [50**2, 100**2],
                       [100**2, 200**2], [200**2, 300**2], [300**2, 1e5**2]]
        area_counts = []
        for area_name, area_range in zip(area_names[1:], area_ranges[1:]):
            area_count = 0
            for i in range(self.num_images):
                if candidate_boxes is None:
                    # default is use the non-gt boxes from roidb
                    non_gt_inds = np.where(roidb[i]['gt_classes'] == 0)[0]
                    boxes = roidb[i]['boxes'][non_gt_inds, :]
                else:
                    boxes = candidate_boxes[i]
                boxes_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
                valid_range_inds = np.where((boxes_areas >= area_range[0]) & (boxes_areas < area_range[1]))[0]
                area_count += len(valid_range_inds)
            area_counts.append(area_count)
        total_counts = float(sum(area_counts))
        for area_name, area_count in zip(area_names[1:], area_counts):
            log_info = 'percentage of {} {}'.format(area_name, area_count / total_counts)
            print log_info
            all_log_info += log_info
        log_info = 'average number of proposal {}'.format(total_counts / self.num_images)
        print log_info
        all_log_info += log_info
        for area_name, area_range in zip(area_names, area_ranges):
            gt_overlaps = np.zeros(0)
            num_pos = 0
            for i in range(self.num_images):
                # check for max_overlaps == 1 avoids including crowd annotations
                max_gt_overlaps = roidb[i]['gt_overlaps'].max(axis=1)
                gt_inds = np.where((roidb[i]['gt_classes'] > 0) & (max_gt_overlaps == 1))[0]
                gt_boxes = roidb[i]['boxes'][gt_inds, :]
                gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
                valid_gt_inds = np.where((gt_areas >= area_range[0]) & (gt_areas < area_range[1]))[0]
                gt_boxes = gt_boxes[valid_gt_inds, :]
                num_pos += len(valid_gt_inds)

                if candidate_boxes is None:
                    # default is use the non-gt boxes from roidb
                    non_gt_inds = np.where(roidb[i]['gt_classes'] == 0)[0]
                    boxes = roidb[i]['boxes'][non_gt_inds, :]
                else:
                    boxes = candidate_boxes[i]
                if boxes.shape[0] == 0:
                    continue

                overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

                _gt_overlaps = np.zeros((gt_boxes.shape[0]))
                # choose whatever is smaller to iterate
                rounds = min(boxes.shape[0], gt_boxes.shape[0])
                for j in range(rounds):
                    # find which proposal maximally covers each gt box
                    argmax_overlaps = overlaps.argmax(axis=0)
                    # get the IoU amount of coverage for each gt box
                    max_overlaps = overlaps.max(axis=0)
                    # find which gt box is covered by most IoU
                    gt_ind = max_overlaps.argmax()
                    gt_ovr = max_overlaps.max()
                    assert (gt_ovr >= 0), '%s\n%s\n%s' % (boxes, gt_boxes, overlaps)
                    # find the proposal box that covers the best covered gt box
                    box_ind = argmax_overlaps[gt_ind]
                    # record the IoU coverage of this gt box
                    _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                    assert (_gt_overlaps[j] == gt_ovr)
                    # mark the proposal box and the gt box as used
                    overlaps[box_ind, :] = -1
                    overlaps[:, gt_ind] = -1
                # append recorded IoU coverage level
                gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

            gt_overlaps = np.sort(gt_overlaps)
            if thresholds is None:
                step = 0.05
                thresholds = np.arange(0.5, 0.95 + 1e-5, step)
            recalls = np.zeros_like(thresholds)

            # compute recall for each IoU threshold
            for i, t in enumerate(thresholds):
                recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
            ar = recalls.mean()

            # print results
            log_info = 'average recall for {}: {:.3f}'.format(area_name, ar)
            print log_info
            all_log_info += log_info
            for threshold, recall in zip(thresholds, recalls):
                log_info = 'recall @{:.2f}: {:.3f}'.format(threshold, recall)
                print log_info
                all_log_info += log_info

        return all_log_info

    @staticmethod
    def merge_roidbs(a, b):
        """
        merge roidbs into one
        :param a: roidb to be merged into
        :param b: roidb to be merged
        :return: merged imdb
        """
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'], b[i]['gt_classes']))
            a[i]['gt_overlaps'] = np.vstack((a[i]['gt_overlaps'], b[i]['gt_overlaps']))
            a[i]['max_classes'] = np.hstack((a[i]['max_classes'], b[i]['max_classes']))
            a[i]['max_overlaps'] = np.hstack((a[i]['max_overlaps'], b[i]['max_overlaps']))
        return a
