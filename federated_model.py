# Copyright 2021, Lanping-Tech.

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models, losses, metrics
import tensorflow_federated as tff

def create_resnet50(input_shape, num_classes):
	conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
	model = models.Sequential()
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.Dense(num_classes, activation='softmax'))
	return model

def create_keras_model_vgg16(input_shape, num_classes):
	conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
	# for layer in conv_base.layers:
	# 	layer.trainable = False
	model = models.Sequential()
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.Dense(num_classes, activation='softmax'))
	return model

def model_select(model_name, input_shape, n_class):
	keras_model = create_keras_model_vgg16(input_shape, n_class) if model_name == 'vgg16' else create_resnet50(input_shape, n_class)
	return keras_model

def get_federated_model_from_keras(model_name, input_shape, input_spec, n_class):
	return tff.learning.from_keras_model(
		model_select(model_name, input_shape, n_class),
		input_spec=input_spec,
		loss=losses.SparseCategoricalCrossentropy(),
		metrics=[metrics.SparseCategoricalAccuracy()])