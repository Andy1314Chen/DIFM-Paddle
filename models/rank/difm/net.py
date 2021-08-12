# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as Fun
import math


class FMLayer(nn.Layer):
    def __init__(self):
        super(FMLayer, self).__init__()
        self.bias = paddle.create_parameter(is_bias=True,
                                            shape=[1],
                                            dtype='float32')

    def forward(self, first_order, dense_features, sparse_features):
        """
        first_order: FM first order (batch_size, 1)
        dense_features: FM dense features (batch_size, embedding_size)
        sparse_features: FM sparse features (batch_size, field_num, embedding_size)
        """
        # (batch_size, 1, embedding_size)
        dense_features = paddle.unsqueeze(dense_features, axis=1)

        # (batch_size, (1 + field_num), embedding_size)
        combined_features = paddle.concat([dense_features, sparse_features], axis=1)

        # sum square part
        # (batch_size, embedding_size)
        summed_features_emb = paddle.sum(combined_features, axis=1)
        summed_features_emb_square = paddle.square(summed_features_emb)

        # square sum part
        squared_features_emb = paddle.square(combined_features)
        # (batch_size, embedding_size)
        squared_sum_features_emb = paddle.sum(squared_features_emb, axis=1)

        # (batch_size, 1)
        logits = first_order + 0.5 * paddle.sum(summed_features_emb_square - squared_sum_features_emb, axis=1,
                                                keepdim=True) + self.bias
        return Fun.sigmoid(logits)




