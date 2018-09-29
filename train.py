#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : ZhangQiang
# @Time    : 9/19/18


import tensorflow as tf

from model import cifar_net

slim = tf.contrib.slim

TF_RECORD = '/mnt/disk0/Data/cifar10/tf_record/train.tfrecord'
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 0.001


# tf.enable_eager_execution()
# tf.executing_eagerly()        # => True


def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([3 * 32 * 32])

    # Reshape from [height * width * depth] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # Custom preprocessing.
    # image = self.preprocess(image)

    return image, label


dataset = tf.data.TFRecordDataset(TF_RECORD)
dataset = dataset.map(parser)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(TRAIN_BATCH_SIZE)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()

next_example, next_label = iterator.get_next()
tf.summary.image('image', next_example, max_outputs=3)

# init = tf.global_variables_initializer()

logits, end_points = cifar_net.cifarnet(next_example, num_classes=10, is_training=True)
tf.summary.image('conv1', end_points['conv1'][:, :, :, 0:1])


loss = tf.losses.sparse_softmax_cross_entropy(next_label, logits)
tf.summary.scalar('loss', loss)
# loss = tf.reduce_mean(loss)

training_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('tmp/train', sess.graph)

finetune = 0
saver = tf.train.Saver()
global_step = 0
with tf.Session() as sess:
    with tf.summary.FileWriter("tmp", sess.graph) as writer:

        if finetune:
            saver.restore(sess, "tmp/model.ckpt")
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        for i in range(100):
            print(i)
            sess.run(iterator.initializer)
            while True:
                try:
                    _, _loss, _merged = sess.run((training_op, loss, merged))

                    if (global_step % 100) == 0:
                        writer.add_summary(_merged, global_step)
                        print("global_step = %d, loss = %f" % (global_step, _loss))

                    if (global_step % 1000) == 0:
                        save_path = saver.save(sess, "tmp/model.ckpt")
                        print("Model saved in path: %s" % save_path)

                    global_step += 1

                except tf.errors.OutOfRangeError:
                    break
