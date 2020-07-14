from fvcore.common.config import CfgNode as CN


_C = CN()

_C.TASK = 'classification'
_C.TRAIN_DIR = ('data/train', )
_C.VAL_DIR = ('data/val',)
_C.OUTPUT_DIR = 'baseline'
_C.EPOCH = 60
_C.BATCH_SIZE = 64
_C.MIXED_PRECISION = False
_C.QUANTIZATION_TRAINING = False
_C.TENSORBOARD = True

_C.MODEL = CN()
_C.MODEL.NAME = 'resnet50'
_C.MODEL.NUM_CLASSES = 3
_C.MODEL.CLASSES = (None,)

_C.MODEL.TEMPERATURE_SCALING = 1

_C.MODEL.AUTOML = False
_C.MODEL.AUTOML_TRIALS = 1000

_C.SOLVER = CN()
_C.SOLVER.NAME = 'sgd'
_C.SOLVER.LR = 0.0002
_C.SOLVER.WEIGHT_DECAY = 0.00001

_C.SOLVER.SCHEDULER = CN()
_C.SOLVER.SCHEDULER.NAME = 'exponential'
_C.SOLVER.SCHEDULER.EPOCHS = (3,)
_C.SOLVER.SCHEDULER.GAMMA = 0.1

_C.DATA = CN()
_C.DATA.SIZE = (299, 299) #w, h
_C.DATA.SHUFFLE_SIZE = 10
_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]

_C.DATA.RANDOM_CROP = False
_C.DATA.RANDOM_CROP_SIZE = (299, 299)

_C.DATA.RANDOM_BRIGHTNESS = False
_C.DATA.RANDOM_BRIGHTNESS_DELTA = 0.5

_C.DATA.RANDOM_CONTRAST = False
_C.DATA.RANDOM_CONTRAST_RANGE = (0.1, 0.1)

_C.DATA.RANDOM_FLIP_LR = True
_C.DATA.RANDOM_FLIP_UD = False

_C.DATA.RANDOM_HUE = False
_C.DATA.RANDOM_HUE_DELTA = 0.3

_C.DATA.RANDOM_SATURATE = False
_C.DATA.RANDOM_SATURATE_RANGE = (0.1, 0.1)