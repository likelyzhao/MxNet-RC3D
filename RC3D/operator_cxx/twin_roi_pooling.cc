/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file psroi_pooling.cc
 * \brief psroi pooling operator
 * \author Yi Li, Tairui Chen, Guodong Zhang, Haozhi Qi, Jifeng Dai
*/
#include "./twin_roi_pooling-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
template<typename Dtype>
inline void TWinROIPoolForward(const Tensor<cpu, 4, Dtype> &out,
                           const Tensor<cpu, 5, Dtype> &data,
                           const Tensor<cpu, 2, Dtype> &bbox,
                           const float temporal_scale_,
                           const int output_dim_,
                           const int group_size_) {


  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
//  Dtype *argmax_data = max_idx.dptr_;
  const int channels_ = data.size(1);
  const int time_length = data.size(2);
  const int height_ = data.size(3);
  const int width_ = data.size(4);

  const int pooled_height_ = out.size(2);
  const int pooled_width_ = out.size(3);

  const int num_rois = bbox.size(0);
  const int data_size = data.size(1) * data.size(2) * data.size(3) * data.size(4);
  const int data_size_c = data.size(2) * data.size(3)* data.size(4);

  const int out_size_c = out.size(2) * out.size(3);
  const int out_size = channels_ * out_size_c;
//  const int max_idx_size_c = max_idx.size(2) * max_idx.size(3);
//  const int max_idx_size = channels_ * max_idx_size_c;
  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    // Increment ROI data pointer
    const Dtype *bottom_rois_n = bottom_rois + n * bbox.size(1);
    Dtype *top_data_n = top_data + n * out_size;
//    Dtype *argmax_data_n = argmax_data + n * max_idx_size;
    int pooled_length_ = 1;
    int roi_batch_ind = bottom_rois_n[0];
    int roi_start_w = 0;
    int roi_start_h = 0;
    int roi_end_w = data.size(3) - 1;
    int roi_end_h = data.size(4) - 1;
    int roi_start_time = bottom_rois_n[1] * temporal_scale_;
    int roi_end_time = bottom_rois_n[2] * temporal_scale_;

    assert(roi_batch_ind >= 0);
    assert(static_cast<index_t>(roi_batch_ind) < data.size(0) /* batch size */);

    // force malformed ROIs to be 1 * 1
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_time_length = max(roi_end_time - roi_start_time + 1, 1);

    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype bin_size_l = static_cast<Dtype>(roi_time_length)
                         / static_cast<Dtype>(pooled_length_);

    const Dtype* batch_data = bottom_data + data_size * roi_batch_ind;

    #pragma omp parallel for
    for (int c = 0; c < channels_; ++c) {
      // Increment all data pointers
      const Dtype* batch_data_c = batch_data + c * data_size_c;
      Dtype* top_data_c = top_data_n + c * out_size_c;
//      Dtype* argmax_data_c = argmax_data_n + c * max_idx_size_c;
      for (int pl = 0; pl < pooled_length_; ++pl) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              // Compute pooling region for this output unit:
              // start (included) = floor(ph * roi_height / pooled_height_)
              // end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
              int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                                  * bin_size_h));
              int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                                  * bin_size_w));
              int lstart = static_cast<int>(floor(static_cast<Dtype>(pl)
                                      * bin_size_l));

              int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                               * bin_size_h));
              int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                               * bin_size_w));
              int lend = static_cast<int>(ceil(static_cast<Dtype>(pl + 1)
                                             * bin_size_l));

              lstart = min(max(lstart + roi_start_time, 0), roi_time_length);
              lend = min(max(lend + roi_start_time, 0), roi_time_length);

              hstart = min(max(hstart + roi_start_h, 0), height_);
              hend = min(max(hend + roi_start_h, 0), height_);
              wstart = min(max(wstart + roi_start_w, 0), width_);
              wend = min(max(wend + roi_start_w, 0), width_);

              bool is_empty = (hend <= hstart) || (wend <= wstart) || (lend <= lstart);

              const int pool_index = (pl * pooled_length_ + ph) * pooled_width_ + pw;
              if (is_empty) {
                top_data_c[pool_index] = 0;
//                argmax_data_c[pool_index] = -1;
              }
              for (int l = lstart; l < lend; ++l) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      const int index = (l * roi_time_length + h) * width_ + w;
                      if (batch_data_c[index] > top_data_c[pool_index]) {
                        top_data_c[pool_index] = batch_data_c[index];
//                        argmax_data_c[pool_index] = index;
                      }
                    }
                  }
              }
            }
          }
      }
    }
  }
  return;

  // NOT_IMPLEMENTED;
  return;
}

template<typename DType>
inline void TWinROIPoolBackwardAcc(const Tensor<cpu, 4, DType> &in_grad,
                            const Tensor<cpu, 5, DType> &out_grad,
                            const Tensor<cpu, 2, DType> &bbox,
                            const float spatial_scale_,
                            const int output_dim_,
                            const int group_size_) {





  // NOT_IMPLEMENTED;
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(TWinROIPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new TWinROIPoolingOp<cpu, DType>(param);
  });
  return op;
}

Operator *TWinROIPoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(TWinROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_TWinROIPooling, TWinROIPoolingProp)
.describe("Performs region-of-interest pooling on inputs. Resize bounding box coordinates by "
"spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled "
"by max pooling to a fixed size output indicated by pooled_size. batch_size will change to "
"the number of region bounding boxes after TWinROIPooling")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 5D Feature maps")
.add_argument("rois", "Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners "
"of designated region of interest. batch_index indicates the index of corresponding image "
"in the input data")
.add_arguments(TWinROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
