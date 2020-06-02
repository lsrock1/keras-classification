from classification.solver.builder import build_optimizer
from .kerasmodels import make_keras_model, keras_factory
from .automl import make_automl_model
from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model
# from autokeras.utils import data_utils


class ModelInterface:
    def __init__(self, args):
        self.args = args
        self.is_automl = args.MODEL.AUTOML
        self.model = self.build(args)
        self.compile()
        if self.args.QUANTIZATION_TRAINING:
            assert not self.is_automl, "autokeras doesn't support quantization aware training"
            self.model = quantize_model(self.model)
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
            self.model.compile(
                optimizer = build_optimizer(self.args),
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy']
            )
            # print(self.model)

    def load_weights(self, path):
        if not self.is_automl:
            self.model.load_weights(path)
        else:
            self.model.export().load_weights(path)

        return self.model

    def fit(self, train, steps_per_epoch, validation_data, use_multiprocessing, workers, epochs, callbacks):
        if self.is_automl:
            self.__automl_fit(dataset=train, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
        else:
            self.__fit(train, steps_per_epoch, validation_data, use_multiprocessing, workers, epochs, callbacks)

    def __automl_fit(self, dataset, epochs, callbacks, validation_data, **kwargs):
        # self.model.inputs[0].shape = [self.args.DATA.SIZE[1], self.args.DATA.SIZE[0], 3]

        # self.model.outputs[0].shape = [self.args.MODEL.NUM_CLASSES]
        # print(self.model.outputs[0].__dict__)
        # print(self.model.outputs)
        # self.model.fit(x=dataset, y=None,
        #                   epochs=epochs,
        #                   callbacks=callbacks,
        #                   validation_data=validation_data,
        #                   fit_on_val_data=False,
        #                   **kwargs)
        # self.__check_data_format(dataset, validation_data)
        # dataset = self.__process_xy(dataset, fit=True)
        # self.model._split_dataset = False
        # x_val = validation_data
        # y_val = None
        # validation_data = self.__process_xy(x_val, validation=True)
        # dataset = dataset.batch(self.args.BATCH_SIZE, drop_remainder=True)
        # validation_data = validation_data.batch(self.args.BATCH_SIZE, drop_remainder=True)
        # print(type(dataset))
        self.model.fit(x=dataset, y=None,
                          epochs=epochs,
                          callbacks=callbacks,
                        #   validation_data=validation_data,
                        #   fit_on_val_data=False,
                          **kwargs)
        # self.model.tuner.search(x=dataset,
        #                   epochs=epochs,
        #                   callbacks=callbacks,
        #                   validation_data=validation_data,
        #                   fit_on_val_data=False,
        #                   **kwargs)

    def __fit(self, train, steps_per_epoch, validation_data, use_multiprocessing, workers, epochs, callbacks, **kwargs):
        self.model.fit(
            train,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            epochs=epochs,
            callbacks=callbacks
        )

    # For AutoKeras
    def __check_data_format(self, x, validation_data):
        x_shapes, y_shapes = tf.compat.v1.data.get_output_shapes(x)
        x_shapes = nest.flatten(x_shapes)
        y_shapes = nest.flatten(y_shapes)
        if len(x_shapes) != len(self.model.inputs):
            raise ValueError(
                'Expect x{in_val} to have {input_num} arrays, '
                'but got {data_num}'.format(
                    in_val=in_val,
                    input_num=len(self.model.inputs),
                    data_num=len(x_shapes)))
        if len(y_shapes) != len(self.model.outputs):
            raise ValueError(
                'Expect y{in_val} to have {output_num} arrays, '
                'but got {data_num}'.format(
                    in_val=in_val,
                    output_num=len(self.model.outputs),
                    data_num=len(y_shapes)))

    def __process_xy(self, x, fit=False, validation=False, predict=False):
        dataset = x
        if not predict:
            y = dataset.map(lambda a, b: b)
            y = [y.map(lambda *a: nest.flatten(a)[index])
                    for index in range(len(self.model.outputs))]
            x = dataset.map(lambda a, b: a)
        x = [x.map(lambda *a: nest.flatten(a)[index])
                for index in range(len(self.model.inputs))]
        # print(self.model.__dict__.keys())
        x = self.__adapt(x, fit, self.model.inputs, self.model._input_adapters)

        if not predict:
            y = self.__adapt(y, fit, self.model._heads, self.model._output_adapters)

        if not predict:
            return tf.data.Dataset.zip((x, y))
        
        if len(self.model.inputs) == 1:
            return x
        
        return x.map(lambda *x: (x, ))

    def __adapt(self, sources, fit, hms, adapters):
        sources = nest.flatten(sources)
        adapted = []
        for source, hm, adapter in zip(sources, hms, adapters):
            if fit:
                source = adapter.fit_transform(source)
                hm.config_from_adapter(adapter)
            else:
                source = adapter.transform(source)
            adapted.append(source)
        if len(adapted) == 1:
            return adapted[0]
        return tf.data.Dataset.zip(tuple(adapted))

def build_compiled_model(cfg):
    return ModelInterface(cfg)
