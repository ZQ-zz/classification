#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : ZhangQiang
# @Time    : 9/19/18


import os

import tensorflow as tf

DATASET_PATH = '/mnt/disk0/Data/cifar10/cifar-10-batches-py/'
TFRECORD_PATH = '/mnt/disk0/Data/cifar10/tf_record/'
# ['data_batch_0', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TRAIN_FILE_NAMES = ['data_batch_%d' % i for i in xrange(1, 5+1)]
TEST_FILE_NAMES = ['test_batch']


# cifar10 dataset python interface
def unpickle(file_name):
    import pickle
    with open(file_name, 'rb') as fo:
        data_dict = pickle.load(fo)  # encoding='bytes'
    return data_dict


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#
def convert_to_tfrecord(input_files, output_file):
    print('convert %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = unpickle(input_file)
            image_data = data_dict['data']
            labels = data_dict['labels']
            for i in range(len(labels)):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(image_data[i].tobytes()),
                        'label': _int64_feature(labels[i])
                    }))
                record_writer.write(example.SerializeToString())


def main():
    # get label info
    info_dict = unpickle(DATASET_PATH + 'batches.meta')
    label_map = info_dict['label_names']

    # train tfrecord
    input_files = []
    output_file = os.path.join(TFRECORD_PATH, 'train.tfrecord')
    for file_name in TRAIN_FILE_NAMES:
        input_file = os.path.join(DATASET_PATH, file_name)
        input_files.append(input_file)
    convert_to_tfrecord(input_files, output_file)

    # test tfrecord
    input_files = []
    output_file = os.path.join(TFRECORD_PATH, 'test.tfrecord')
    for file_name in TEST_FILE_NAMES:
        input_file = os.path.join(DATASET_PATH, file_name)
        input_files.append(input_file)
    convert_to_tfrecord(input_files, output_file)


if __name__ == '__main__':
    main()





