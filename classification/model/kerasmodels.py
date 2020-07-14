from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Activation
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


def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple."""
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])


class CustomModel(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        model_name = cfg.MODEL.NAME
        task_name = cfg.TASK
        num_classes = cfg.MODEL.NUM_CLASSES
        size = [cfg.DATA.SIZE[1], cfg.DATA.SIZE[0]]
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        classes = cfg.MODEL.CLASSES
        assert model_name in keras_factory
        self.unk = 'unknown' in classes
        if self.unk:
            self.unk_idx = classes.index('unknown')
        self.cfg = cfg
        self.model = keras_factory[model_name](
            weights='imagenet', include_top=False, input_tensor=Input(shape= size + [3]))
        self.fc = tf.keras.Sequential([
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(1024, activation='relu'),
            Dense(num_classes, dtype='float32')
        ])
        self.softmax = Activation('softmax')
        self.add_weight_decay(weight_decay)

    def export(self):
        output = self.model.output
        output = self.fc(output)
        output = self.softmax(output)
        return Model(inputs=self.model.input, outputs=output)

    def add_weight_decay(self, weight_decay):
        for layer in self.model.layers:
            layer.trainable = True
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)
        
        for layer in self.fc.layers:
            layer.trainable = True
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)

    def __call__(self, x, training=False):
        self.model.trainable = training
        self.fc.trainable = training
        output = self.model(x)
        output = self.fc(output)
        output = self.softmax(output)
        return output

    def train_step(self, data):
        data, label, weights = unpack_x_y_sample_weight(data)
            
        with tf.GradientTape() as tape:
            pred = self(data, True)

            if self.unk:
                unk_index = tf.math.equal(label, tf.constant(self.unk_idx, dtype=tf.int64))
                cls_index = tf.math.logical_not(unk_index)
                cls_pred = tf.boolean_mask(pred, cls_index)
                cls_gt = tf.boolean_mask(label, cls_index)
                unk_pred = tf.boolean_mask(pred, unk_index)
                cls_loss = tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(cls_gt, cls_pred))
                # tf.print(tf.shape(cls_loss))
                unk_loss = 0.5 * -tf.math.reduce_mean(tf.math.reduce_mean(unk_pred, axis=1) - tf.reduce_logsumexp(unk_pred, axis=1))
                # total_loss = tf.constant(0, dtype=tf.int64)
                # total_loss = cls_loss + unk_loss
                cls_loss = tf.cond(tf.math.is_nan(cls_loss), lambda: tf.constant(0, dtype=tf.float32) * cls_loss, lambda: cls_loss)
                unk_loss = tf.cond(tf.math.is_nan(unk_loss), lambda: tf.constant(0, dtype=tf.float32) * unk_loss, lambda: unk_loss)
                total_loss = cls_loss + unk_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss,
                    "cls_loss": cls_loss,
                    "unk_loss": unk_loss,
                }
            else:
                total_loss = keras.losses.sparse_categorical_crossentropy(data, label)
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss
                }

def make_keras_model(task_name, model_name, num_classes, size, weight_decay):
    assert model_name in keras_factory

    input_tensor = Input(shape=size + (3,))
    model = keras_factory[model_name](weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    if task_name == 'classification':
        x = Dropout(0.3)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(num_classes, dtype='float32')(x)
        x = Activation('softmax')(x)
    model = Model(inputs=model.input, outputs=x)

    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer= l2(weight_decay)

    model.summary()
    return model
