from classification.solver.builder import build_optimizer
from .kerasmodels import make_keras_model, keras_factory


def get_model(cfg):
    if cfg.MODEL.NAME in keras_factory:
        return make_keras_model(cfg.MODEL.NAME,
                                cfg.MODEL.NUM_CLASSES,
                                (cfg.DATA.SIZE[1], cfg.DATA.SIZE[0]),
                                cfg.SOLVER.WEIGHT_DECAY)


def build_compiled_model(cfg):
    model = get_model(cfg)
    model.compile(
        optimizer = build_optimizer(cfg),
        loss='sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model
