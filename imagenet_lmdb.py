# coding: utf-8
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from PIL import Image
import six
import lmdb
import pyarrow as pa
import torch.utils.data as data
# PATHS can be filled with the default data path, or be specified by the argument
PATHs = []

def check_path(paths):
    for path in paths:
        if osp.exists(path):
            return path
    return None

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class ImageNetLMDB(data.Dataset):
    def __init__(self, root, list_file, ignore_label=True):
        root = check_path([root,]+PATHs)
        db_path = osp.join(root, list_file)
        self.db_path = db_path
        self.env = None
        if 'train_1percent' in self.db_path:
            self.length = 12811
        elif 'train_10percent' in self.db_path:
            self.length = 128116
        elif 'train' in self.db_path:
            self.length = 1281167
        elif 'val' in self.db_path:
            self.length = 50000
        else:
            raise NotImplementedError
        self.ignore_label = ignore_label

    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.kys = loads_pyarrow(txn.get(b'__keys__'))

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()

        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.kys[index])
        unpacked = loads_pyarrow(byteflow)

        # load img.
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        # load label.
        target = unpacked[1]

        if self.ignore_label:
            return img
        else:
            return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    def get_length(self):
        return self.length

    def get_sample(self, idx):
        return self.__getitem__(idx)

