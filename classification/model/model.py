from classification.solver.builder import build_optimizer
from .kerasmodels import make_keras_model, keras_factory
from .automl import make_automl_model


class ModelInterface:
    def __init__(self, args):
        self.args = args
        self.is_automl = args.MODEL.AUTOML
        self.model = self.build(args)
        self.compile()

    def build(self, args):
        return self.get_model(args)

    def get_model(self, cfg):
        if cfg.MODEL.NAME in keras_factory and not cfg.MODEL.AUTOML:
            return make_keras_model(cfg.MODEL.NAME,
                                    cfg.MODEL.NUM_CLASSES,
                                    (cfg.DATA.SIZE[1], cfg.DATA.SIZE[0]),
                                    cfg.SOLVER.WEIGHT_DECAY)
        else:
            return make_automl_model(cfg)
    
    def compile(self):
        if not self.is_automl:
            self.model = self.model.compile(
                optimizer = build_optimizer(self.args),
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy']
            )

    def fit(self, train, steps_per_epoch, validation_data, use_multiprocessing, workers, epochs, callbacks):
        if self.is_automl:
            self.__automl_fit(dataset=train, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
        else:
            self.__fit(train, steps_per_epoch, validation_data, use_multiprocessing, workers, epochs, callbacks)

    def __automl_fit(self, dataset, epochs, callbacks, validation_data, **kwargs):
        # self.model.inputs[0].shape = [self.args.DATA.SIZE[1], self.args.DATA.SIZE[0], 3]

        # self.model.outputs[0].shape = [self.args.MODEL.NUM_CLASSES]
        # print(self.model.outputs[0].__dict__)
        self.model.fit(x=dataset,
        y=None,
                          epochs=epochs,
                          callbacks=callbacks,
                          validation_data=validation_data,
                          fit_on_val_data=False,
                          **kwargs)

    def __fit(self, train, steps_per_epoch, validation_data, use_multiprocessing, workers, epochs, callbacks, **kwargs):
        self.model.fit(
            data.train_tfrecords,
            steps_per_epoch=data.train_length,
            validation_data=data.val_tfrecords,
            use_multiprocessing=True,
            workers=6,
            epochs=cfg.EPOCH,
            callbacks=build_callbacks(cfg)
        )


def build_compiled_model(cfg):
    return ModelInterface(cfg)
