#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import random
import numpy as np

from paddle.io import IterableDataset
# from tools.utils.utils_single import load_yaml


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        random.shuffle(file_list)
        self.file_list = file_list
        self.init()

    def init(self):
        from operator import mul
        padding = 0
        sparse_slots = "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots + self.dense_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                lines = rf.readlines()
                random.shuffle(lines)
                for l in lines:
                    line = l.strip().split(" ")
                    output = [(i, []) for i in self.slots]
                    for i in line:
                        slot_feasign = i.split(":")
                        slot = slot_feasign[0]
                        if slot not in self.slots:
                            continue
                        if slot in self.sparse_slots:
                            feasign = int(slot_feasign[1])
                        else:
                            feasign = float(slot_feasign[1])
                        output[self.slot2index[slot]][1].append(feasign)
                        self.visit[slot] = True
                    for i in self.visit:
                        slot = i
                        if not self.visit[slot]:
                            if i in self.dense_slots:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding] *
                                    self.dense_slots_shape[self.slot2index[i]])
                            else:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding])
                        else:
                            self.visit[slot] = False
                    # sparse
                    output_list = []
                    for key, value in output[:-1]:
                        output_list.append(np.array(value).astype('int64'))
                    # dense
                    output_list.append(
                        np.array(output[-1][1]).astype("float32"))
                    # list
                    yield output_list
                    # [label,sparse_col_1,sparse_col_2,...,sparse_col_26,[dense_array]]


# if __name__ == "__main__":
#     """
#         test criteo dataset
#     """
#     config = load_yaml("config.yaml")
#     dataset = RecDataset(file_list=["data/sample_data/train/sample_train.txt"],
#                          config=config)
#     for i in dataset:
#         print(i)
#         break
#
#     """
#     [array([0]), array([737395]), array([210498]), array([903564]), array([286224]), array([286835]), array([906818]),
#      array([906116]), array([67180]), array([27346]), array([51086]), array([142177]), array([95024]), array([157883]),
#       array([873363]), array([600281]), array([812592]), array([228085]), array([35900]), array([880474]), array([984402]),
#        array([100885]), array([26235]), array([410878]), array([798162]), array([499868]), array([306163]),
#         array([0.        , 0.00497512, 0.05      , 0.08      , 0.20742187,    0.028     , 0.35      , 0.08      , 0.082
#         , 0.        ,0.4       , 0.        , 0.08      ], dtype=float32)]
#     """
