import mxnet as mx
from config import config
from utils.symbol import Symbol
from operator_py.proposal_twin import *
from operator_py.proposal_target_twin import *

class RC3D(Symbol):
	def __init__(self):
		"""
		Use __init__ to define parameter network needs
		"""
		self.eps = 1e-5
#		self.num_anchors = config.num_anchors
		self.use_global_stats = True
		self.workspace = 512
		self.units = (3, 4, 23, 3)  # use for 101
		self.filter_list = [256, 512, 1024, 2048]

	def get_symbol(self, cfg, is_train=True):

		num_anchors = cfg.network.num_anchors
		num_classes = cfg.dataset.NUM_CLASSES

		input_data = mx.symbol.Variable(name="data")

		gt_boxes = mx.symbol.Variable(name="gt_windows")
		rpn_label = mx.symbol.Variable(name='label')
		rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
		rpn_bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight')
 #		rpn_bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight')

		###  conv1
		conv1 = mx.symbol.Convolution(data=input_data, kernel=(3, 3, 3), pad=(1, 1, 1), num_filter=64, name="conv1a")
		relu1 = mx.symbol.Activation(data=conv1, act_type="relu", name="relu1a")
		pool1 = mx.symbol.Pooling(
			data=relu1, pool_type="max", kernel=(1, 2, 2), stride=(1, 2, 2), name="pool1")

		### conv2
		conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3, 3), pad=(1, 1, 1), num_filter=128, name="conv2a")
		relu2 = mx.symbol.Activation(data=conv2, act_type="relu", name="relu2a")
		pool2 = mx.symbol.Pooling(
			data=relu2, pool_type="max", kernel=(2, 2, 2), stride=(2, 2, 2), name="pool2")

		### conv3

		conv3_1 = mx.symbol.Convolution(data=pool2, kernel=(3, 3, 3), pad=(1, 1, 1), num_filter=256, name="conv3a")
		relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3a")


		conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3, 3, 3), pad=(1, 1, 1), num_filter=256, name="conv3b")
		relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3b")
		pool3 = mx.symbol.Pooling(
			data=relu3_2, pool_type="max", kernel=(2, 2, 2), stride=(2, 2, 2), name="pool3")

		#### conv4

		conv4_1 = mx.symbol.Convolution(data=pool3, kernel=(3, 3, 3), pad=(1, 1, 1), num_filter=512, name="conv4a")
		relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4a")


		conv4_2 = mx.symbol.Convolution(data=relu4_1, kernel=(3, 3, 3), pad=(1, 1, 1), num_filter=512, name="conv4b")
		relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4b")
		pool4 = mx.symbol.Pooling(
			data=relu4_2, pool_type="max", kernel=(2, 2, 2), stride=(2, 2, 2), name="pool4")

		### conv5
		conv5_1 = mx.symbol.Convolution(data=pool4, kernel=(3, 3, 3), pad=(1, 1, 1), num_filter=512, name="conv5a")
		relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5a")


		conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3, 3), pad=(1, 1, 1), num_filter=512, name="conv5b")
		relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5b")

		### rpn_conv/3x3 proposal subnet
		rpn_conv_3x3_1 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(1,2,2), num_filter=512, name="rpn_conv/3x3")
		rpn_relu_1 = mx.symbol.Activation(data=rpn_conv_3x3_1, act_type="relu", name="rpn_relu/3x3")

		rpn_conv_3x3_2 = mx.symbol.Convolution(data=rpn_relu_1, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(1,2,2), num_filter=512, name="rpn_conv/3x3_2")
		rpn_relu_2 = mx.symbol.Activation(data=rpn_conv_3x3_2, act_type="relu", name="rpn_relu/3x3_2")

		rpn_output = mx.symbol.Pooling(
			data=rpn_relu_2, pool_type="max", kernel=(1, 2, 2), name="rpn/output_pool")

		rpn_cls_score = mx.symbol.Convolution(data=rpn_output, kernel=(1, 1, 1), num_filter=2*num_anchors, name="rpn_cls_score")

		rpn_bbox_pred = mx.symbol.Convolution(data=rpn_output, kernel=(1, 1, 1), num_filter=2*num_anchors, name="rpn_twin_pred")

		# bounding box regression
		rpn_bbox_loss_ = rpn_bbox_inside_weight * mx.symbol.smooth_l1(name='rpn_loss_twin', scalar=3.0,
		                                                       data=(rpn_bbox_pred - rpn_bbox_target))
		rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
		                                grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

		# prepare rpn data
		rpn_cls_score_reshape = mx.symbol.Reshape(
			data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")


#		rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
#		                                       normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")

		rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
		                                       normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")


		### classification subnet

		rpn_cls_act = mx.symbol.SoftmaxActivation(
			data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
		rpn_cls_act_reshape = mx.symbol.Reshape(
			data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')


#		rois = mx.contrib.symbol.Proposal(
#			cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred,op_type='proposal_twin',
#			feature_stride=cfg.RPN_FEAT_STRIDE, scales=tuple(cfg.ANCHOR_SCALES), ratios=tuple(cfg.ANCHOR_RATIOS),
#			rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
#			threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

#		proposal_twin

		rois = mx.symbol.Custom(
			cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred,op_type='proposal_twin',
			feat_stride=cfg.network.TWIN_STRIDE, scales=tuple(cfg.network.TWIN_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
			rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
			threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE,name='proposal_twin')

		gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 3), name='gt_boxes_reshape')

		group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target_twin',
		                         num_classes=num_classes, batch_images=cfg.TRAIN.BATCH_IMAGES,
		                         batch_rois=cfg.TRAIN.BATCH_ROIS, fg_fraction=cfg.TRAIN.FG_FRACTION,cfg=cfg)

		rois = group[0]
		label = group[1]
		bbox_target = group[2]
		bbox_weight = group[3]

		pool5 = mx.symbol.ROIPooling(
			name='roi_pool5', data=relu5_2, rois=rois, pooled_size=(4, 4), spatial_scale=1.0 / cfg.network.TWIN_STRIDE)

		flatten = mx.symbol.Flatten(data=pool5, name="flatten")
		fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
		relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
		drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")

		cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop6, num_hidden=num_classes)
		cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
		# bounding box regression
		bbox_pred = mx.symbol.FullyConnected(name='twin_pred', data=drop6, num_hidden=num_classes * 2)

		bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
		bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)

		# reshape output
		label = mx.symbol.Reshape(data=label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
		cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
		                             name='cls_prob_reshape')
		bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_classes),
		                              name='bbox_loss_reshape')

		group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])

		self.sym = group
		return group
