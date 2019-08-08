import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboard_logger as tbrd
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from collections import Counter

from model import LeNet, reweight_autodiff


mnist = input_data.read_data_sets(train_dir='mnist', one_hot=False)

num_classes = 10
dbg_steps = 20


def prepare_data(corruption_matrix, gold_fraction=0.05, merge_valset=True):
    np.random.seed(1)

    mnist_images = np.copy(mnist.train.images)
    mnist_labels = np.copy(mnist.train.labels)
    if merge_valset:
        mnist_images = np.concatenate([mnist_images, np.copy(mnist.validation.images)], axis=0)
        mnist_labels = np.concatenate([mnist_labels, np.copy(mnist.validation.labels)])

    indices = np.arange(len(mnist_labels))
    np.random.shuffle(indices)

    mnist_images = mnist_images[indices]
    mnist_labels = mnist_labels[indices].astype(np.long)
    mnist_labels_orig = np.copy(mnist_labels)

    num_gold = int(len(mnist_labels)*gold_fraction)
    num_silver = len(mnist_labels) - num_gold

    for i in range(num_silver):
        mnist_labels[i] = np.random.choice(num_classes, p=corruption_matrix[mnist_labels[i]])

    # dtype flag is important to the DataSet class doesn't renormalize the images by /255
    gold = DataSet(mnist_images[num_silver:], mnist_labels[num_silver:], reshape=False, dtype=dtypes.uint8)
    silver = DataSet(mnist_images[:num_silver], np.array(list(zip(mnist_labels[:num_silver], mnist_labels_orig[:num_silver]))), reshape=False, dtype=dtypes.uint8)

    return gold, silver


def uniform_mix_C(mixing_ratio):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(1)

    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def train_and_test(flags, corruption_level=0, gold_fraction=0.5, get_C=uniform_mix_C):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    C = get_C(corruption_level)

    gold, silver = prepare_data(C, gold_fraction)

    print("Gold shape = {}, Silver shape = {}".format(gold.images.shape, silver.images.shape))

    # TODO : test on whole set
    test_x = torch.from_numpy(mnist.test.images[:500].reshape([-1, 1, 28, 28]))
    test_y = torch.from_numpy(mnist.test.labels[:500]).type(torch.LongTensor)
    print("Test shape = {}".format(test_x.shape))

    model = LeNet()
    optimizer = torch.optim.Adam([p for p in model.parameters()], lr=0.001)

    for step in range(flags.num_steps) :
        x, y = silver.next_batch(flags.batch_size)
        y, y_true = np.array([l[0] for l in y]), np.array([l[1] for l in y])
        x_val, y_val = gold.next_batch(min(flags.batch_size, flags.nval))

        x, y = torch.from_numpy(x.reshape([-1, 1, 28, 28])), torch.from_numpy(y).type(torch.LongTensor)
        x_val, y_val = torch.from_numpy(x_val.reshape([-1, 1, 28, 28])), torch.from_numpy(y_val).type(torch.LongTensor)

        # forward
        if flags.method == "l2w" :
            ex_wts = reweight_autodiff(model, x, y, x_val, y_val)
            logits, loss = model.loss(x, y, ex_wts)

            if step % dbg_steps == 0 :
                tbrd.log_histogram("ex_wts", ex_wts, step=step)
                tbrd.log_value("More_than_0.01", sum([x > 0.01 for x in ex_wts]), step=step)
                tbrd.log_value("More_than_0.05", sum([x > 0.05 for x in ex_wts]), step=step)
                tbrd.log_value("More_than_0.1",  sum([x > 0.1  for x in ex_wts]), step=step)

                mean_on_clean_labels = np.mean([ex_wts[i] for i in range(len(y)) if y[i] == y_true[i]])
                mean_on_dirty_labels = np.mean([ex_wts[i] for i in range(len(y)) if y[i] != y_true[i]])
                tbrd.log_value("mean_on_clean_labels", mean_on_clean_labels, step=step)
                tbrd.log_value("mean_on_dirty_labels", mean_on_dirty_labels, step=step)
        else :
            logits, loss = model.loss(x, y)

        print("Loss = {}".format(loss))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tbrd.log_value("loss", loss, step=step)

        if step % dbg_steps == 0 :
            model.eval()

            pred = torch.max(model.forward(test_x), 1)[1]
            test_acc = torch.sum(torch.eq(pred, test_y)).item() / float(test_y.shape[0])
            model.train()

            print("Test acc = {}.".format(test_acc))
            tbrd.log_value("test_acc", test_acc, step=step)


def main(flags) :
    tbrd.configure(flags.model_dir)
    corruption_fnctn = uniform_mix_C if flags.corruption_type == 'uniform_mix' else flip_labels_C

    gold_fraction = 0.05
    corruption_level = 0.9

    train_and_test(flags, corruption_level, gold_fraction, corruption_fnctn)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model_dir")
    parser.add_argument("--corruption_type", default="flip_labels", type=str, choices=["uniform_mix", "flip_labels"])
    parser.add_argument("--num_steps", default=1000, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--nval", default=40, type=int)
    parser.add_argument("--method", default="l2w", type=str, choices=["l2w", "baseline"])

    args = parser.parse_args()
    print(args)

    main(args)
