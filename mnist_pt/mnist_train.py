import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from collections import Counter

from model import LeNet, reweight_autodiff


mnist = input_data.read_data_sets(train_dir='mnist', one_hot=False)
num_classes = 10
dbg_steps = 20


def prepare_data(corruption_matrix, gold_fraction=0.5, merge_valset=True):
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

    num_gold = int(len(mnist_labels)*gold_fraction)
    num_silver = len(mnist_labels) - num_gold

    for i in range(num_silver):
        mnist_labels[i] = np.random.choice(num_classes, p=corruption_matrix[mnist_labels[i]])

    dataset = {'x': mnist_images, 'y': mnist_labels}
    gold = DataSet(dataset['x'][num_silver:], dataset['y'][num_silver:], reshape=False)
    silver = DataSet(dataset['x'][:num_silver], dataset['y'][:num_silver], reshape=False)

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
    test_x = torch.from_numpy(mnist.test.images[:200].reshape([-1, 1, 28, 28]))
    test_y = torch.from_numpy(mnist.test.labels[:200]).type(torch.LongTensor)
    print("Test shape = {}".format(test_x.shape))

    model = LeNet()
    optimizer = torch.optim.SGD([p for p in model.parameters()], lr=0.1, momentum = 0.9, weight_decay=1e-6)

    for step in range(flags.num_steps) :
        x, y = silver.next_batch(flags.batch_size)
        x_val, y_val = gold.next_batch(min(flags.batch_size, flags.nval))

        x, y = torch.from_numpy(x.reshape([-1, 1, 28, 28])), torch.from_numpy(y)
        x_val, y_val = torch.from_numpy(x_val.reshape([-1, 1, 28, 28])), torch.from_numpy(y_val)

        if step % dbg_steps == 0 :
            model.eval()

            pred = torch.max(model.forward(x_val), 1)[1]
            old_val_acc = torch.sum(torch.eq(pred, y_val)).item() / float(y_val.shape[0])

            pred = torch.max(model.forward(test_x), 1)[1]
            old_test_acc = torch.sum(torch.eq(pred, test_y)).item() / float(test_y.shape[0])
            model.train()

        # get training example weights
        ex_wts = reweight_autodiff(model, x, y, x_val, y_val)

        # forward
        logits, loss = model.loss(x, y, ex_wts)
        print("Loss = {}".format(loss))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % dbg_steps == 0 :
            model.eval()

            pred = torch.max(model.forward(x_val), 1)[1]
            new_val_acc = torch.sum(torch.eq(pred, y_val)).item() / float(y_val.shape[0])

            pred = torch.max(model.forward(test_x), 1)[1]
            new_test_acc = torch.sum(torch.eq(pred, test_y)).item() / float(test_y.shape[0])
            model.train()

            print("Old val = {}, New val = {}. Old test = {}, New test = {}.".format(old_val_acc, new_val_acc, old_test_acc, new_test_acc))


def main(flags) :
    corruption_fnctn = uniform_mix_C if flags.corruption_type == 'uniform_mix' else flip_labels_C

    gold_fraction = 0.05
    corruption_level = 0.3

    train_and_test(flags, corruption_level, gold_fraction, corruption_fnctn)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption_type", default="flip_labels", type=str, choices=["uniform_mix", "flip_labels"])
    parser.add_argument("--num_steps", default=1000, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--nval", default=40, type=int)

    args = parser.parse_args()

    main(args)
