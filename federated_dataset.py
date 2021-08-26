# Copyright 2021, Lanping-Tech.

import os
import collections
from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import reshape, nest, config
import tensorflow_federated as tff

def load_numpy_data(data_path):
	x_path = os.path.join(data_path,'x.npy')
	y_path = os.path.join(data_path,'y.npy')
	X = np.load(x_path)
	Y = np.load(y_path)
	index = [i for i in range(len(X))]
	shuffle(index)
	X = X[index, :, :, :]
	Y = Y[index]
	X = X[:1000, :, :, :]
	Y = Y[:1000]
	X = X.astype(np.float32)
	Y = Y.astype(np.float32)
	return X, Y

def get_client_train_dataset(data, label, n_clients):
	image_per_set = int(np.floor(len(data) / n_clients))
	client_train_dataset = collections.OrderedDict()
	for i in range(1, n_clients+1):
		client_name = "client_" + str(i)
		start = image_per_set * (i-1)
		end = image_per_set * i

		print(f"Adding data from {start} to {end} for client : {client_name}")
		client_dataset = collections.OrderedDict((('label', label[start:end]), ('pixels', data[start:end])))
		client_train_dataset[client_name] = client_dataset
	return client_train_dataset, image_per_set

def data_preprocess(dataset, n_epochs, shuffle_buffer, batch_size, crop_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE):

	def batch_format_fn(element):
		images = tf.image.resize_with_crop_or_pad(element['pixels'], target_height=crop_shape, target_width=crop_shape)
		images = tf.image.per_image_standardization(images)
		return collections.OrderedDict(x=images, y=reshape(element['label'], [-1, 1]))

	return dataset.repeat(n_epochs).shuffle(shuffle_buffer).batch(batch_size).map(batch_format_fn, num_parallel_calls)

def load_federated_data(data_path, n_clients, n_epochs, batch_size, crop_shape):
	X, Y = load_numpy_data(data_path)
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

	client_train_dataset, shuffle_buffer = get_client_train_dataset(x_train, y_train, n_clients)

	train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

	federated_train_data = [data_preprocess(train_dataset.create_tf_dataset_for_client(client_id), n_epochs, shuffle_buffer, batch_size, crop_shape) 
											for client_id in train_dataset.client_ids]

	x_test = tf.image.resize_with_crop_or_pad(x_test, target_height=crop_shape, target_width=crop_shape)
	x_test = tf.image.per_image_standardization(x_test)
	y_test = reshape(y_test, [-1, 1])

	return federated_train_data, x_test, y_test, len(np.unique(y_train))