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


class MLPLayer(nn.Layer):
    def __init__(self, input_shape, units_list=None, l2=0.01, last_action=None, **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.units_list = units_list
        self.l2 = l2
        self.mlp = []
        self.last_action = last_action

        for i, unit in enumerate(units_list[:-1]):
            if i != len(units_list) - 1:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.ParamAttr(
                                             initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)
                self.add_sublayer('relu_%d' % i, relu)

                norm = paddle.nn.BatchNorm1D(units_list[i + 1])
                self.mlp.append(norm)
                self.add_sublayer('norm_%d' % i, norm)
            else:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit)))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                if last_action is not None:
                    relu = paddle.nn.ReLU()
                    self.mlp.append(relu)
                    self.add_sublayer('relu_%d' % i, relu)

    def forward(self, inputs):
        outputs = inputs
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
        return outputs


class FENLayer(nn.Layer):
    def __init__(self,
                 sparse_field_num,
                 sparse_feature_num,
                 sparse_feature_dim,
                 dense_feature_dim,
                 fen_layers_size):
        super(FENLayer, self).__init__()
        self.sparse_field_num = sparse_field_num
        self.sparse_feature_num = sparse_feature_num
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.fen_layers_size = fen_layers_size

        self.fen_mlp = MLPLayer(input_shape=(sparse_field_num + 1) * sparse_feature_dim,
                                units_list=fen_layers_size)

        self.sparse_embedding = ;

    def forward(self, dense_inputs, sparse_inputs):
        pass


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





