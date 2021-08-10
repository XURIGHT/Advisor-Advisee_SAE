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


class DataSet(object):
    def __init__(self,
                   datas,
                   labels,
                   contexts,
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
            assert datas.shape[0] == labels.shape[0] and datas.shape[0] == contexts.shape[0], (
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
        self._contexts = contexts
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def datas(self):
        return self._datas


    @property
    def labels(self):
        return self._labels.reshape([-1, 2])

    @property
    def contexts(self):
        return self._contexts

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._datas = self.datas[perm0]
            self._labels = self.labels[perm0]
            self._contexts = self.contexts[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            datas_rest_part = self._datas[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            context_rest_part = self._contexts[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._datas = self.datas[perm]
                self._labels = self.labels[perm]
                self._contexts = self.contexts[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            datas_new_part = self._datas[start:end]
            labels_new_part = self._labels[start:end]
            context_new_part = self._contexts[start:end]

            return numpy.concatenate((datas_rest_part, datas_new_part), axis=0), \
                   numpy.concatenate((labels_rest_part, labels_new_part), axis=0).reshape([-1, 2]), \
                   numpy.concatenate((context_rest_part, context_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._datas[start:end], self._labels[start:end].reshape([-1, 2]), self._contexts[start:end]


def shuffle(train_data, train_labels, train_contexts):
    train_data_ = np.empty(shape=[0, train_data.shape[1]], dtype=np.float64)
    train_labels_ = np.empty(shape=[0, train_labels.shape[1]], dtype=np.float64)
    train_contexts_ = np.empty(shape=[0, train_contexts.shape[1]], dtype=np.float64)

    perm = list(np.random.permutation(train_data.shape[0]))
    for i in perm:
        vec1 = np.zeros((1, train_data.shape[1]), dtype=np.float64)
        for j in range(train_data.shape[1]):
            vec1[0][j] = train_data[i][j]
        train_data_ = np.append(train_data_, vec1, axis=0)

        vec2 = np.zeros((1, train_labels.shape[1]), dtype=np.float64)
        for j in range(train_labels.shape[1]):
            vec2[0][j] = train_labels[i][j]
        train_labels_ = np.append(train_labels_, vec2, axis=0)

        vec3 = np.zeros((1, train_contexts.shape[1]), dtype=np.float64)
        for j in range(train_contexts.shape[1]):
            vec3[0][j] = train_contexts[i][j]
        train_contexts_ = np.append(train_contexts_, vec3, axis=0)

    return train_data_, train_labels_, train_contexts_


def read_data_sets(
        train_dir,
        context_dir,
        test_factor=0.1,
        dtype=dtypes.float32,
        reshape=False,
        validation_size=100,
        seed=None):

    index2context = {}
    with open(context_dir, encoding="utf-8") as f:
        cnt = 0
        n_fea = 100
        vec = np.zeros((1, n_fea), dtype=np.float64)
        for line in f:
            cnt += 1
            if cnt == 1:
                index = int(float(line.strip()))
            else:
                content = line.strip().split('\t')
                for i in range(len(content)):
                    vec[0][i] = float(content[i])
                index2context[index] = vec
                vec = np.zeros((1, n_fea), dtype=np.float64)
                cnt = 0
        f.close()

    n_year = 52
    n_class = 2
    train_data = np.empty(shape=[0, n_year * 5], dtype=np.float64)
    train_labels = np.empty(shape=[0, n_class], dtype=np.float64)
    train_contexts = np.empty(shape=[0, n_fea * 2], dtype=np.float64)

    with open(train_dir, encoding="utf-8") as f:
        cnt = 0
        vec = np.zeros((1, n_year * 5), dtype=np.float64)
        for line in f:
            content = line.strip().split('\t')
            cnt += 1

            if cnt == 1:
                vec_ = np.zeros((1, n_fea * 2), dtype=np.float64)
                content = line.strip().split('\t')
                index1 = int(content[0])
                index2 = int(content[1])
                if index2context.__contains__(index1):
                    for i in range(n_fea):
                        vec_[0][i] = index2context[index1][0][i]
                else:
                    print(index1)
                if index2context.__contains__(index2):
                    for i in range(n_fea):
                        vec_[0][i + n_fea] = index2context[index2][0][i]
                else:
                    print(index2)
                train_contexts = np.append(train_contexts, vec_, axis=0)


            elif cnt < 6:
                for i in range(n_year):
                    vec[0][i + (cnt - 2) * n_year] = float(content[i])

            elif cnt == 6:
                for i in range(n_year):
                    vec[0][i + (cnt - 2) * n_year] = float(content[i])
                train_data = np.append(train_data, vec, axis=0)
                vec = np.zeros((1, n_year * 5), dtype=np.float64)

            elif cnt == 7:
                tmp = np.zeros((1, n_class), dtype=np.float64)
                if int(content[-1]) == 0:
                    tmp[0][0] = 1
                else:
                    tmp[0][1] = 1
                train_labels = np.append(train_labels, tmp, axis=0)
                cnt = 0
        f.close()

    mu = np.mean(train_data, axis=0)
    sigma = np.std(train_data, axis=0, ddof=1)
    train_data = np.nan_to_num((train_data - mu) / sigma)

    train_data, train_labels, train_contexts = shuffle(train_data, train_labels, train_contexts)

    num_test = int(test_factor * train_data.shape[0])

    validation_data = train_data[:validation_size]
    validation_labels = train_labels[:validation_size]
    validation_contexts = train_contexts[:validation_size]

    test_data = train_data[-num_test:]
    test_labels = train_labels[-num_test:]
    test_contexts = train_contexts[-num_test:]


    train_data = train_data[validation_size:-num_test]
    train_labels = train_labels[validation_size:-num_test]
    train_contexts = train_contexts[validation_size:-num_test]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_data, train_labels, train_contexts, **options)
    validation = DataSet(validation_data, validation_labels, validation_contexts, **options)
    test = DataSet(test_data, test_labels, test_contexts, **options)

    return base.Datasets(train=train, validation=validation, test=test)


if __name__ == "__main__":
    dataset = read_data_sets("../dataset/features_matrix.txt", "../dataset/context_feature_after_sae.txt")
