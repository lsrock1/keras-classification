from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201


keras_factory = {
    'mobilenet': MobileNetV2,
    'resnet50': ResNet50V2,
    'resnet101': ResNet101V2,
    "resnet152": ResNet152V2,
    'inceptionv3': InceptionV3,
    'inception-resnet': InceptionResNetV2,
    'densenet121': DenseNet121,
    'densenet169': DenseNet169,
    'densenet201': DenseNet201
    # 'efficientnetb0': "https://tfhub.dev/google/efficientnet/b0/feature_vector/1",
    # 'efficientnetb1': "https://tfhub.dev/google/efficientnet/b1/feature_vector/1",
    # 'efficientnetb2': "https://tfhub.dev/google/efficientnet/b2/feature_vector/1"
}


def make_keras_model(model_name, num_classes, size, weight_decay):
    assert model_name in keras_factory

    input_tensor = Input(shape=size + (3,))
    model = keras_factory[model_name](weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=x)

    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer= l2(weight_decay)

    model.summary()
    return model
