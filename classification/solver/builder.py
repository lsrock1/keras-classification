import tensorflow as tf

from .scheduler import build_scheduler

__all__ = ['sgd', 'adam']


def SGD(scheduler):
    return tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9, nesterov=True)


def ADAM(scheduler):
    return tf.keras.optimizers.Adam(learning_rate=scheduler)


__pair = {
    'sgd': SGD,
    'adam': ADAM
}

def build_optimizer(cfg):
    assert cfg.SOLVER.NAME in __pair
    scheduler = build_scheduler(cfg)
    return __pair[cfg.SOLVER.NAME](scheduler)
