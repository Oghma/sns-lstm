#!/usr/bin/env python

import os
import math
import numpy as np

if not os.path.exists("datasets/processed"):
    os.makedirs("datasets/processed/train")
    os.makedirs("datasets/processed/test")
    os.makedirs("datasets/processed/val")

train_folder = "datasets/raw/train"
val_folder = "datasets/raw/val"
test_folder = "datasets/raw/test"

train_files = [
    f for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))
]
val_files = [
    f for f in os.listdir(val_folder) if os.path.isfile(os.path.join(val_folder, f))
]
test_files = [
    f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))
]

for i in range(len(train_files)):
    train = np.loadtxt(os.path.join(train_folder, train_files[i]))
    val = np.loadtxt(os.path.join(val_folder, val_files[i]))

    minX = np.concatenate((train[:, 2], val[:, 2])).min()
    minY = np.concatenate((train[:, 3], val[:, 3])).max()
    maxX = np.concatenate((train[:, 2], val[:, 2])).max()
    maxY = np.concatenate((train[:, 3], val[:, 3])).max()

    diag = math.sqrt((maxX - minX) ** 2 + (maxY - minY) ** 2)

    train[:, 2] = (train[:, 2] - minX) / diag
    train[:, 3] = (train[:, 3] - minY) / diag
    val[:, 2] = (val[:, 2] - minX) / diag
    val[:, 3] = (val[:, 3] - minY) / diag

    np.savetxt("datasets/processed/train/" + train_files[i], train, delimiter="\t")
    np.savetxt("datasets/processed/val/" + val_files[i], val, delimiter="\t")

for data in test_files:
    test = np.loadtxt(os.path.join(test_folder, data))

    minX = min(test[:, 2])
    minY = min(test[:, 3])
    maxX = max(test[:, 2])
    maxY = max(test[:, 3])

    diag = math.sqrt((maxX - minX) ** 2 + (maxY - minY) ** 2)

    test[:, 2] = (test[:, 2] - minX) / diag
    test[:, 3] = (test[:, 3] - minY) / diag

    np.savetxt("datasets/processed/test/" + data, test, delimiter="\t")

print("All datasets normalized")
