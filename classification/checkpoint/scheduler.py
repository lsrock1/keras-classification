import tensorflow as tf
import bisect


__all__ = ['exponential', 'fixed', 'polynomial']


def exponential(lr, decay_epochs, gamma):
    def scheduler(epoch):
        return lr * (gamma ** bisect.bisect_right(decay_epochs, epoch))
    
    return tf.keras.callbacks.LearningRateScheduler(scheduler)


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
     # * tf.math.exp(0.1 * (10 - epoch))

    return pair[cfg.SOLVER.SCHEDULER.NAME](cfg.SOLVER.LR, cfg.SOLVER.SCHEDULER.EPOCHS, cfg.SOLVER.SCHEDULER.GAMMA)