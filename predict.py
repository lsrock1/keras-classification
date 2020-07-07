import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from classification.model import build_compiled_model
from classification.datasets.dataset import build_data
from classification.configs import cfg

import argparse
from pathlib import Path


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
    model = model.load_weights(latest)
    model = model.export()

    return model


def predict_and_show(model, ):
    val_dir = cfg.VAL_DIR
    val_dir = Path(val_dir)
    class_names = [d.basename() for d in val_dir.glob('*')]
    val_images = val_dir.glob('*/*')
    
    if cfg.NUM_CLASSES != len(class_names):
        pass    
    
    for image_path in val_images:
        gt_name = image_path.dirname()
        image_raw = tf.io.read_file(str(image_path))
        image = tf.image.decode_image(image_raw)
        results = model.predict(image)