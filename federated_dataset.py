# Copyright 2021, Lanping-Tech.

import os
import collections
import random
from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import tensorflow as tf
from tensorflow import reshape, nest, config
import tensorflow_federated as tff

from utils.utils_paths import list_images
import cv2



def load_numpy_data(data_path='./dataset', crop_shape=128):
	imagePaths = sorted(list(list_images(data_path)))
	random.seed(2021)
	random.shuffle(imagePaths)
	
	data = []
	labels = []
	for imagePath in imagePaths:

		image = cv2.imread(imagePath)
		image = cv2.resize(image, (crop_shape, crop_shape))
		data.append(image)
		# 读取标签
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	# 对图像数据做scale操作
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	return data, labels


def get_client_train_dataset(data, label, n_clients):
    image_per_set = int(np.floor(len(data) / n_clients))
    client_train_dataset = collections.OrderedDict()
    for i in range(1, n_clients+1):
        client_name = "client_" + str(i)
        start = image_per_set * (i-1)
        end = image_per_set * i

        print(f"Adding data from {start} to {end} for client : {client_name}")
        client_dataset = collections.OrderedDict(
            (('label', label[start:end]), ('pixels', data[start:end])))
        client_train_dataset[client_name] = client_dataset
    return client_train_dataset, image_per_set


def get_client_test_dataset(data, label, n_clients):
    image_per_set = int(np.floor(len(data) / n_clients))
    print('image_per_set:', image_per_set)
    client_test_dataset = collections.OrderedDict()
    for i in range(1, n_clients+1):
        client_name = "client_" + str(i)
        start = image_per_set * (i-1)
        end = image_per_set * i

        print(f"Adding data from {start} to {end} for client : {client_name}")
        client_dataset = collections.OrderedDict(
            (('label', label[start:end]), ('pixels', data[start:end])))
        client_test_dataset[client_name] = client_dataset
    return client_test_dataset, image_per_set


def data_preprocess(dataset, n_epochs, shuffle_buffer, batch_size, crop_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE):

    def batch_format_fn(element):
        images = tf.image.resize_with_crop_or_pad(
            element['pixels'], target_height=crop_shape, target_width=crop_shape)
        images = tf.image.per_image_standardization(images)
        return collections.OrderedDict(x=images, y=element['label'])

    return dataset.repeat(n_epochs).shuffle(shuffle_buffer).batch(batch_size).map(batch_format_fn, num_parallel_calls)


def load_federated_data(data_path, n_clients, n_epochs, batch_size, crop_shape):
    X, Y = load_numpy_data(data_path, crop_shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    # print(y_train.shape)
    y_test = lb.transform(y_test)

    client_train_dataset, shuffle_buffer = get_client_train_dataset(
        x_train, y_train, n_clients)

    train_dataset = tff.simulation.datasets.TestClientData(
        client_train_dataset)

    federated_train_data = [data_preprocess(train_dataset.create_tf_dataset_for_client(client_id), n_epochs, shuffle_buffer, batch_size, crop_shape) for client_id in train_dataset.client_ids]

    client_test_dataset, shuffle_buffer = get_client_test_dataset(
        x_test, y_test, n_clients)
    test_dataset = tff.simulation.datasets.TestClientData(
        client_test_dataset)
    federated_test_data = [data_preprocess(test_dataset.create_tf_dataset_for_client(client_id), 1, shuffle_buffer, batch_size, crop_shape)
                           for client_id in test_dataset.client_ids]

    return federated_train_data, federated_test_data, y_train.shape[1]
