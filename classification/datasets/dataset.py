from .transforms import DataAugmenter
import threading

import numpy as np
from glob import glob
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
# from imblearn.keras import BalancedBatchGenerator, balanced_batch_generator
# from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


AUTOTUNE = tf.data.experimental.AUTOTUNE
# CLASS_NAMES = ['unknown', 'person', 'lie_down', ]


class Dataprocessor:
    def __init__(self, args):
        self.args = args
        self.classes = self.read_classes(args.TRAIN_DIR)
        self.total_train = self.make_tfrecord(args.TRAIN_DIR)
        self.total_val = self.make_tfrecord(args.VAL_DIR)
        self.train_tfrecords = self.load_tfrecord(args.TRAIN_DIR)
        self.train_tfrecords = self.create_dataset(self.train_tfrecords)
        self.val_tfrecords = self.load_tfrecord(args.VAL_DIR)
        self.val_tfrecords = self.create_dataset(self.val_tfrecords, True)

    @property
    def train_length(self):
        return self.total_train // self.args.BATCH_SIZE

    @property
    def val_length(self):
        return self.total_val // self.args.BATCH_SIZE

    def create_dataset(self, tfrecords, is_val=False):
        augmentation = DataAugmenter(self.args, is_val) 
        return tfrecords.cache()\
                        .shuffle(self.args.DATA.SHUFFLE_SIZE)\
                        .map(augmentation)\
                        .batch(self.args.BATCH_SIZE)\
                        .prefetch(buffer_size=AUTOTUNE)

    def read_classes(self, paths):
        print(f'read classes from data path: {paths} ..')
        dirs = set()
        for path in paths:
            classes = [os.path.basename(f) for f in glob(os.path.join(path, '*')) if os.path.isdir(f)]
            # print(classes)
            dirs = dirs.union(set(classes))
        
        dirs = sorted(list(dirs))
        assert len(dirs) > 1, f'only {len(dirs)} class exists!'
        print(f'{len(dirs)} classes exist')
        for name in dirs:
            print(name)
        
        print('Loading class finished!')
        return dirs

    def make_tfrecord(self, paths):
        lengths = []
        for path in paths:
            record_file = os.path.join(path, 'data.tfrecords')
            if os.path.exists(record_file):
                with open(record_file + '.length', 'r') as l:
                    lengths.append(int(l.readline()))
                continue
            
            files = [f for f in glob(os.path.join(path, '*/*')) if f.endswith('jpg') or f.endswith('.png')]
            # print(os.path.join(path, '*/*.{jpg,png}'))
            # print(files)
            print('making tfrecords..')
            # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
            length = len(files)
            with tf.io.TFRecordWriter(record_file) as writer:
                for f in tqdm(files):
                    image_string = open(f, 'rb').read()
                    label = os.path.basename(os.path.dirname(f))
                    tf_example = self.__make_feature_from(image_string, label)
                    writer.write(tf_example.SerializeToString())
            lengths.append(length)
            with open(record_file + '.length', 'w') as l:
                l.write(str(length))

        return sum(lengths)

    def __make_feature_from(self, image_string, label):
        label = self.classes.index(label)
        image_shape = tf.image.decode_jpeg(image_string).shape
        if isinstance(image_string, type(tf.constant(0))):
            image_string = image_string.numpy()
        h, w, _ = image_shape
        feature = {
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def load_tfrecord(self, paths):
        records = []
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        record_paths = [os.path.join(path, 'data.tfrecords') for path in paths]
        dataset = tf.data.TFRecordDataset(record_paths)

        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, image_feature_description)
            # image = tf.io.decode_raw(example['image'], tf.uint8)
            image = tf.image.decode_image(example['image'], channels=3)
            image = tf.reshape(image, [example['height'], example['width'], 3])
            return image, example['label']

        return dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)


# def image_resize(image, size):
#     width, height = size
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     origin_width = width
#     origin_height = height
#     dim = None
#     (h, w) = image.shape[:2]

#     if h > w:
#         width = None
#     else:
#         height = None

#     # if both the width and height are None, then return the
#     # original image
#     # if width is None and height is None:
#     #     return image

#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)

#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))
#     # resize the image
#     resized = cv2.resize(image, dim)

#     # return the resized image
#     resized = pad_size(resized, origin_width, origin_height)
#     return resized


# def pad_size(image, width, height):
#     (h, w) = image.shape[:2]
#     # print('vertical: ', (height - h), ' horizontal: ', width-w)
#     top, bottom = (height - h) // 2, (height - h) - ((height - h) // 2)
#     left, right = (width - w) // 2, (width - w) - ((width - w) // 2)
#     # print((height - h), ' ', 0, ' ', (width - w), ' ')
#     # print('top: ', top, ' bottom: ', bottom, ' left: ', left, ' right: ', right)
#     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
#     return image


# def decode_img(img, mean, std, size):
#     # print('*'*50)
#     # print(img)
#     # convert the compressed string to a 3D uint8 tensor
#     img = cv2.imread(img[0])#.astype(float)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = image_resize(img, size).astype(float)
#     img = (img - mean) / std

#     # print(img.shape)
#     # img = tf.io.read_file(img[0])
#     # img = tf.image.decode_jpeg(img, channels=3)
#     # # Use `convert_image_dtype` to convert to floats in the [0,1] range.
#     # img = tf.image.convert_image_dtype(img, tf.float32).eval()
#     # resize the image to the desired size.
#     # print(img.shape)
#     # print('8'*80)
#     return img


# def to_onehot(index):
#     result = np.zeros([len(CLASS_NAMES)])
#     result[index] = 1
#     return result


# def data_ready(path):
#     images = sorted(glob(os.path.join(path, '*/*.jpg')))
#     return np.array(images).reshape(-1, 1), np.array([CLASS_NAMES.index(i.split('/')[-2]) for i in images])


# class ValGenerator(Sequence):
#     def __init__(self, x, y, mean, std, size):
#         self.lock = threading.Lock()
#         self.x = x
#         self.y = y
#         self.datagen = ImageDataGenerator(
#             rescale=1/255)
#         self._shape = x.shape
#         self.mean = np.array(mean).reshape(1, 1, -1)
#         self.std = np.array(std).reshape(1, 1, -1)
#         self.size = size
#         # self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(-1, 1), y.reshape(-1, 1), sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)

#     def __len__(self):
#         return self._shape[0]

#     def __getitem__(self, idx):
#         with self.lock:
#             x_batch, y_batch = self.x[idx:idx+1], self.y[idx: idx+1]
#             x_batch, y_batch = np.stack([decode_img(i, self.mean, self.std, self.size) for i in x_batch]), to_categorical(y_batch.reshape(-1), num_classes=3)
#             # print(x_batch.shape)
#             # x_batch = x_batch.reshape(-1, *self._shape[1:])
#             return self.datagen.flow(x_batch, y_batch, batch_size=1).next()


# class CustomBalancedDataGenerator(Sequence):
#     """ImageDataGenerator + RandomOversampling"""
#     def __init__(self, x, y, batch_size, mean, std, size):
#         self.lock = threading.Lock()
#         self.datagen = ImageDataGenerator(
#             rescale=1/255,
#             # featurewise_center=True,
#             # featurewise_std_normalization=True,
#             rotation_range=3,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             horizontal_flip=True)
#         self.size = size
#         self.batch_size = batch_size
#         self._shape = x.shape
#         self.mean = np.array(mean).reshape(1, 1, -1)
#         self.std = np.array(std).reshape(1, 1, -1)
#         self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(-1, 1), y.reshape(-1, 1), sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)

#     def __len__(self):
#         return self._shape[0] // self.batch_size

#     def __getitem__(self, idx):
#         with self.lock:
#             x_batch, y_batch = self.gen.__next__()
#             x_batch, y_batch = np.stack([decode_img(i, self.mean, self.std, self.size) for i in x_batch]), to_categorical(y_batch.reshape(-1), num_classes=3)
#             return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()

def build_data(cfg):
    return Dataprocessor(cfg)
    # augment = Augmentor(True)
    # X, y = [], []
    # for folder in cfg.TRAIN_DIR:
    #     data = data_ready(folder)
    #     X.append(data[0])
    #     y.append(data[1])
    # X = np.concatenate(X, axis=0)
    # y = np.concatenate(y, axis=0)

    # train_batches = CustomBalancedDataGenerator(
    #     X, y, batch_size=cfg.BATCH_SIZE, mean=cfg.DATA.MEAN, std=cfg.DATA.STD, size=cfg.DATA.SIZE)
    
    # X, y = [], []
    # for folder in cfg.VAL_DIR:
    #     data = data_ready(folder)
    #     X.append(data[0])
    #     y.append(data[1])
    # X = np.concatenate(X, axis=0)
    # y = np.concatenate(y, axis=0)
    # # X, y = data_ready(cfg.VAL_DIR)
    # val_batches = ValGenerator(X, y, mean=cfg.DATA.MEAN, std=cfg.DATA.STD, size=cfg.DATA.SIZE)
    