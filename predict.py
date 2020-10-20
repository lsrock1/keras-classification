import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from classification.model import build_compiled_model
from classification.datasets.dataset import build_data
from classification.configs import cfg

import argparse
from pathlib import Path
import cv2
import numpy as np


def get_model():
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
    
    latest = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
    model = build_compiled_model(cfg)
    model.load_weights(latest)
    model = model.export()

    return model


def put_text(img, texts):
        # You may need to adjust text size and position and size.
        # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
    h = 30
    for text in texts:
        img = cv2.putText(img, str(text), (0, h), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        h += 30

    return img


def predict_and_show(model, ):
    val_dir = ['/home/ocrusr/classification_keras/from_int/']#cfg.VAL_DIR
    val_dir = cfg.VAL_DIR
    val_dir = Path(val_dir[0])
    class_names = [d.name for d in val_dir.glob('*')]
    val_images = val_dir.glob('*/*')
    val_images = list(val_images)
    np.random.shuffle(val_images)
    classes = cfg.MODEL.CLASSES
    
    class_names = ['person', 'falldown', 'unknown']
    corr = 0
    total = 0
    for image_path in val_images:
        
        # if 'unk' in str(image_path):
        #     continue
        gt_name = image_path.parent.name
        gt_label = class_names.index(gt_name)
        # if gt_label == 2:
        #     continue
        total += 1
        image_raw = tf.io.read_file(str(image_path))
        image = tf.image.decode_jpeg(image_raw, dct_method='INTEGER_ACCURATE')
        # print(image)
        # print(type(image))
        # image = tf.image.resize(image, cfg.DATA.SIZE)
        image_ = image / 255
        image_ = image_ - [[cfg.DATA.MEAN]]
        image_ = image_ / [[cfg.DATA.STD]]
        image_ = tf.image.resize_with_pad(image_, cfg.DATA.SIZE[1], cfg.DATA.SIZE[0])
        image_ = np.expand_dims(image_, axis=0)
        results = model.predict(image_)
        results = results[0]
        pred = np.argmax(results)
        pred_value = np.max(results)
        if pred_value < 0.7:
            pred = 2
        print(pred)
        # if pred == gt_label:
        #     corr += 1
        # # if gt_label == 2:
        # #     print(pred_value)
        # else:
        #     print(gt_label, ' but ', pred, ' value: ', pred_value)
        # print(results)
        texts = [cn + ': ' + str(r) for cn, r in zip(['person', 'falldown'], results.tolist())]
        img = put_text(image.numpy().astype(np.uint8), texts)
        cv2.imshow('t', img)
        k = cv2.waitKey(0)
        if k == 27: # esc key
            cv2.destroyAllWindows()
            break
        print(corr/total)
    print(corr/total)


def main():
    model = get_model()
    predict_and_show(model)

if __name__ == '__main__':
    main()
