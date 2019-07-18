import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.examples.tutorials.mnist import input_data
from collections import Counter

from model import LeNet


mnist = input_data.read_data_sets(train_dir='mnist', one_hot=False)
num_classes = 10


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

    dataset = {'x': mnist_images.reshape([-1, 1, 28, 28]), 'y': mnist_labels}
    gold = {'x': dataset['x'][num_silver:], 'y': dataset['y'][num_silver:]}

    return dataset, gold, num_gold, num_silver


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

    dataset, gold, num_gold, num_silver = prepare_data(C, gold_fraction)
    print("Training set shape = {}, num_gold = {}, num_silver = {}".format(dataset["x"].shape, num_gold, num_silver))

    test_x = torch.from_numpy(mnist.test.images[:200].reshape([-1, 1, 28, 28]))
    test_y = torch.from_numpy(mnist.test.labels[:200]).type(torch.LongTensor)
    print("Test set shape = {}".format(test_x.shape))

    model = LeNet()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=1e-6)

    num_batches = num_silver // flags.batch_size

    for epoch in range(flags.epochs) :
        indices = np.arange(num_silver)
        np.random.shuffle(indices)

        for i in range(num_batches) :
            offset = i * flags.batch_size

            batch_x = torch.from_numpy(dataset["x"][indices[offset : offset+flags.batch_size]])
            batch_y = torch.from_numpy(dataset["y"][indices[offset : offset+flags.batch_size]])

            # forward
            logits = model.forward(batch_x)

            # backward
            loss = F.cross_entropy(logits, batch_y)
            print("Loss = {}".format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        pred = torch.max(model.forward(test_x), 1)[1]
        acc = torch.sum(torch.eq(pred, test_y)).item() / float(test_y.shape[0])
        print("Accuracy = {}".format(acc))


def main(flags) :
    corruption_fnctn = uniform_mix_C if flags.corruption_type == 'uniform_mix' else flip_labels_C

    gold_fraction = 0.05
    corruption_level = 0.3

    train_and_test(flags, corruption_level, gold_fraction, corruption_fnctn)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption_type", default="flip_labels", type=str, choices=["uniform_mix", "flip_labels"])
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=100, type=int)

    args = parser.parse_args()

    main(args)
