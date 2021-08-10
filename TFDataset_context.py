# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
from six.moves import xrange
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy as np


class DataSetContext(object):
    def __init__(self,
                   datas,
                   labels,
                   fake_data=False,
                   one_hot=True,
                   dtype=dtypes.float32,
                   reshape=False,
                   seed=None):
        seed1, seed2 = random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot

        else:
            assert datas.shape[0] == labels.shape[0], (
                'datas.shape: %s labels.shape: %s' % (datas.shape, labels.shape))
            self._num_examples = datas.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert datas.shape[3] == 1
                datas = datas.reshape(datas.shape[0],
                                        datas.shape[1] * datas.shape[2])
            if dtype == dtypes.float32:
                datas = datas.astype(numpy.float32)

        self._datas = datas
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def datas(self):
        return self._datas


    @property
    def labels(self):
        return self._labels.reshape([-1, 1])

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._datas = self.datas[perm0]
            self._labels = self.labels[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            datas_rest_part = self._datas[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._datas = self.datas[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            datas_new_part = self._datas[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate(
                  (datas_rest_part, datas_new_part), axis=0), numpy.concatenate(
                      (labels_rest_part, labels_new_part), axis=0).reshape([-1, 1])
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._datas[start:end], self._labels[start:end].reshape([-1, 1])


def read_data_sets(
        train_dir,
        test_factor=0.1,
        fake_data=False,
        one_hot=True,
        dtype=dtypes.float32,
        reshape=False,
        validation_size=0,
        seed=None):
    if fake_data:
        def fake():
            return DataSetContext([], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)
        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    n_year = 52
    n_class = 1
    train_data = np.empty(shape=[0, n_year * 50], dtype=np.float64)
    train_labels = np.empty(shape=[0, n_class], dtype=np.float64)

    with open(train_dir, encoding="utf-8") as f:
        cnt = 0
        for line in f:
            content = line.strip().split('\t')
            cnt += 1

            if cnt == 1:
                tmp = np.zeros((1, n_class), dtype=np.float64)
                tmp[0][0] = int(content[0])
                train_labels = np.append(train_labels, tmp, axis=0)

            elif cnt == 2:
                vec = np.zeros((1, n_year * 50), dtype=np.float64)
                for i in range(n_year * 50):
                    vec[0][i] = content[i]
                train_data = np.append(train_data, vec, axis=0)
                cnt = 0
        f.close()

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSetContext(train_data, train_labels, **options)

    print(train_data.shape)
    print(train_labels.shape)

    return base.Datasets(train=train, validation=None, test=None)


if __name__ == "__main__":
    dataset = read_data_sets("../dataset/author_context_feature.txt")
    a, b = dataset.train.next_batch(64)
    print(a.shape)
    print(b.shape)
    print(dataset)
