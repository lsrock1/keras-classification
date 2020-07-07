import tensorflow as tf
from tensorflow.image import ResizeMethod


class DataAugmenter:
    def __init__(self, args, is_val):
        self.args = args
        self.is_val = is_val

    def __call__(self, image, label=None):
        # 0 ~ 1, pad resize
        image = tf.image.convert_image_dtype(image, tf.float32)
        # image = tf.math.multiply(image, self.args)
        image = tf.image.resize_with_pad(image, self.args.DATA.SIZE[1], self.args.DATA.SIZE[0])
        
        if self.is_val:
            if label != None:
                return image, label
            else:
                return image

        # random crop
        if self.args.DATA.RANDOM_CROP:
            image = tf.image.random_crop(image, self.args.DATA.RANDOM_CROP_SIZE[1], self.args.DATA.RANDOM_CROP_SIZE[0])

        if self.args.DATA.RANDOM_BRIGHTNESS:
            image = tf.image.random_brightness(image, self.args.DATA.RANDOM_BRIGHTNESS_DELTA)

        if self.args.DATA.RANDOM_CONTRAST:
            image = tf.image.random_contrast(image, self.args.DATA.RANDOM_CONTRAST_RANGE[0], self.args.DATA.RANDOM_CONTRAST_RANGE[1])

        if self.args.DATA.RANDOM_FLIP_LR:
            image = tf.image.random_flip_left_right(image)
        
        if self.args.DATA.RANDOM_FLIP_UD:
            image = tf.image.random_flip_up_down(image)

        if self.args.DATA.RANDOM_HUE:
            image = tf.image.random_hue(image, self.args.DATA.RANDOM_HUE_DELTA)

        if self.args.DATA.RANDOM_SATURATE:
            image = tf.image.random_saturation(image, self.args.DATA.RANDOM_SATURATE_RANGE[0], self.args.DATA.RANDOM_SATURATE_RANGE[1])

        if label != None:
            return image, label
        else:
            return image