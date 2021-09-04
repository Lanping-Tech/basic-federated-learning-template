# Copyright 2021, Lanping-Tech.

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models, losses, metrics
import tensorflow_federated as tff

def create_resnet50(input_shape, num_classes):
	model = ResNet50(weights=None, include_top=True, input_shape=input_shape, classes=num_classes)
	return model

def create_keras_model_vgg16(input_shape, num_classes):
	model = VGG16(weights=None, include_top=True, input_shape=input_shape, classes=num_classes)
	return model

def model_select(model_name, input_shape, n_class):
	keras_model = create_keras_model_vgg16(input_shape, n_class) if model_name == 'vgg16' else create_resnet50(input_shape, n_class)
	return keras_model

def get_federated_model_from_keras(model_name, input_shape, input_spec, n_class):
	def model_fn():
		return tff.learning.from_keras_model(
			model_select(model_name, input_shape, n_class),
			input_spec=input_spec,
			loss=losses.CategoricalCrossentropy(),
			metrics=[metrics.CategoricalAccuracy()])
	return model_fn