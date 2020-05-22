import tensorflow as tf


__all__ = ['exponential', 'fixed', 'polynomial']


def exponential(lr, decay_step, gamma):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        lr, decay_step, gamma, staircase=True, name='exponential_decay_learning_rate'
    )


def fixed(lr, decay_step, gamma):
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')


def polynomial(lr, decay_step, gamma):
    return tf.keras.optimizers.schedules.PolynomialDecay(
        lr,
        decay_step,
        0.000000000001,
        power=0.5
    )


pair = {
    'exponential': exponential,
    'fixed': fixed,
    'polynomial': polynomial
}

def build_scheduler(cfg):
    assert cfg.SOLVER.SCHEDULER.NAME in __all__
    return pair[cfg.SOLVER.SCHEDULER.NAME](cfg.SOLVER.LR, cfg.SOLVER.SCHEDULER.STEP, cfg.SOLVER.SCHEDULER.GAMMA)