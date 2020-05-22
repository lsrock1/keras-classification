import tensorflow as tf
import os


def build_checkpoint_callback(cfg):
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    return tf.keras.callbacks.ModelCheckpoint(os.path.join(cfg.OUTPUT_DIR, 'cp-{epoch:02d}-{val_acc:.2f}.ckpt'),
                                                 save_weights_only=True,
                                                 verbose=1)
