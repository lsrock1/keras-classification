from classification.model import build_compiled_model
from classification.datasets.dataset import build_data
from classification.configs import cfg
from classification.callbacks.engine import build_callbacks
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import tensorflow as tf
import argparse
from glob import glob
import os
import cv2
import numpy as np


def __resize_and_padding_zero(image, desired_size=299):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def cv2_imread():
    classes = cfg.MODEL.CLASSES
    files = [f for f in glob(os.path.join(cfg.VAL_DIR[0], '*/*')) if f.endswith('jpg') or f.endswith('.png')]
    # print(files)
    for f in files:
        class_name = os.path.basename(os.path.dirname(f))
        # print(class_name)
        if class_name not in classes:
            continue
        else:
            class_idx = classes.index(class_name)
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = img - [[cfg.DATA.MEAN]]
        img = img / [[cfg.DATA.STD]]
        img = __resize_and_padding_zero(img)
        # print(img.shape)
        img = np.expand_dims(img, axis=0)
        # print(img.shape)
        yield img, np.array([class_idx])


def main():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--restart", default=0, type=int)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)

    # float16, mixed precision
    if cfg.MIXED_PRECISION:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    latest = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
    model = build_compiled_model(cfg)
    model = model.load_weights(latest)
    data = build_data(cfg)

    model.evaluate(
        cv2_imread(),
        use_multiprocessing=False,
        # workers=6,
        callbacks=build_callbacks(cfg)
    )

    # model.evaluate(
    #     data.val_tfrecords,
    #     use_multiprocessing=True,
    #     workers=6,
    #     callbacks=build_callbacks(cfg)
    # )


if __name__ == '__main__':
    main()