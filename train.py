from classification.model import build_compiled_model
from classification.datasets.dataset import build_data
from classification.configs import cfg
from classification.checkpoint.engine import build_checkpoint_callback

import argparse


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
    
    model = build_compiled_model(cfg)
    data = build_data(cfg)
    # print(type(data.train_tfrecords))
    model.fit(
        data.train_tfrecords,
        steps_per_epoch=data.train_length,
        validation_data=data.val_tfrecords,
        # use_multiprocessing=True,
        workers=6,
        epochs=cfg.EPOCH,
        callbacks=[build_checkpoint_callback(cfg)]
    )


if __name__ == '__main__':
    main()