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
                 fen_layers_size,
                 dense_layers_size,
                 batch_size):
        super(FENLayer, self).__init__()
        self.sparse_field_num = sparse_field_num
        self.sparse_feature_num = sparse_feature_num
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.fen_layers_size = fen_layers_size
        self.dense_layers_size = dense_layers_size
        self.batch_size = batch_size

        self.fen_mlp = MLPLayer(input_shape=(sparse_field_num + 1) * sparse_feature_dim,
                                units_list=fen_layers_size)

        self.sparse_embedding = paddle.nn.Embedding(num_embeddings=self.sparse_feature_num,
                                                    embedding_dim=self.sparse_feature_dim,
                                                    sparse=True,
                                                    weight_attr=paddle.ParamAttr(
                                                        name="SparseFeatFactors",
                                                        initializer=paddle.nn.initializer.Uniform()))
        self.sparse_weight = paddle.nn.Embedding(num_embeddings=self.sparse_feature_num,
                                                 embedding_dim=1,
                                                 sparse=True,
                                                 weight_attr=paddle.ParamAttr(
                                                     name="SparseFeatFactors2",
                                                     initializer=paddle.nn.initializer.Uniform()))
        self.dense_linear = paddle.nn.Linear(in_features=self.dense_feature_dim,
                                             out_features=1)
        self.dense_mlp = MLPLayer(input_shape=self.dense_feature_dim,
                                  units_list=self.dense_layers_size,
                                  last_action="relu")

    def forward(self, sparse_inputs, dense_inputs):
        # ------------------ first order ------------------------------------
        # (batch_size, sparse_field_num)
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        # (batch_size, sparse_field_num, 1)
        sparse_emb_one = self.sparse_weight(sparse_inputs_concat)
        # (batch_size, sparse_field_num)
        sparse_emb_one = paddle.squeeze(sparse_emb_one, axis=-1)

        # (batch_size, 1)
        dense_emb_one = self.dense_linear(dense_inputs)

        # (batch_size, (sparse_field_num + 1))
        feat_emb_one = paddle.concat([dense_emb_one, sparse_emb_one], axis=1)

        # -------------------- fen layer ------------------------------------
        # (batch_size, embedding_size)
        dense_embedding = self.dense_mlp(dense_inputs)
        dense_embedding = paddle.unsqueeze(dense_embedding, axis=1)

        # (batch_size, sparse_field_num, embedding_size)
        sparse_embedding = self.sparse_embedding(sparse_inputs_concat)

        # (batch_size, (sparse_field_num + 1), embedding_size)
        feat_embeddings = paddle.concat([dense_embedding, sparse_embedding], axis=1)

        # (batch_size, -1)
        # feat_embeddings = paddle.reshape(feat_embeddings, shape=(self.batch_size, -1))
        # (batch_size, (sparse_field_num + 1))
        m_x = Fun.softmax(self.fen_mlp(paddle.reshape(feat_embeddings, shape=(self.batch_size, -1))))

        # (batch_size, (sparse_field_num + 1))
        feat_emb_one = feat_emb_one * m_x
        # (batch_size, (sparse_field_num + 1), embedding_size)
        feat_embeddings = feat_embeddings * paddle.unsqueeze(m_x, axis=-1)

        # (batch_size, 1)
        first_order = paddle.sum(feat_emb_one, axis=1, keepdim=True)
        return first_order, feat_embeddings


class FMLayer(nn.Layer):
    def __init__(self):
        super(FMLayer, self).__init__()
        self.bias = paddle.create_parameter(is_bias=True,
                                            shape=[1],
                                            dtype='float32')

    def forward(self, first_order, combined_features):
        """
        first_order: FM first order (batch_size, 1)
        combined_features: FM sparse features (batch_size, sparse_field_num + 1, embedding_size)
        """
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


class IFM(nn.Layer):
    def __init__(self,
                 sparse_field_num,
                 sparse_feature_num,
                 sparse_feature_dim,
                 dense_feature_dim,
                 fen_layers_size,
                 dense_layers_size,
                 batch_size):
        super(IFM, self).__init__()
        self.sparse_field_num = sparse_field_num
        self.sparse_feature_num = sparse_feature_num
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.fen_layers_size = fen_layers_size
        self.dense_layers_size = dense_layers_size
        self.batch_size = batch_size
        self.fen_layer = FENLayer(sparse_field_num=self.sparse_field_num,
                                  sparse_feature_num=self.sparse_feature_num,
                                  sparse_feature_dim=self.sparse_feature_dim,
                                  dense_feature_dim=self.dense_feature_dim,
                                  fen_layers_size=self.fen_layers_size,
                                  dense_layers_size=self.dense_layers_size,
                                  batch_size=self.batch_size)
        self.fm_layer = FMLayer()

    def forward(self, sparse_inputs, dense_inputs):
        first_order, combined_features = self.fen_layer(sparse_inputs, dense_inputs)
        return self.fm_layer(first_order, combined_features)

