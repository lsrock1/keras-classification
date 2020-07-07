import tensorflow as tf
import os
import datetime


def build_callbacks(cfg):
    callbacks = []
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(cfg.OUTPUT_DIR, 'cp-{epoch:02d}-{val_accuracy:.2f}.ckpt'),
                                                 save_weights_only=True, save_best_only=True,
                                                 verbose=1)
    callbacks.append(checkpoint)

    if cfg.TENSORBOARD:
        log_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

    return callbacks
