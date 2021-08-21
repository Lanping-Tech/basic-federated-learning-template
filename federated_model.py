from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers, models


def create_resnet50(input_shape, num_classes):
  conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
  model = models.Sequential()
  model.add(conv_base)
  model.add(layers.Flatten())
  model.add(layers.Dense(num_classes, activation='softmax'))
  conv_base.trainable = False
  return model
