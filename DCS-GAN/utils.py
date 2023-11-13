import os
import tensorflow as tf


def create_dir(dir):
    """ Create the directory.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} createrd')
    else:
        print(f'Directory {dir} already exists')  

    return dir

@tf.function
def load_image(image_file, image_size=None):
    """ Load the image file.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_bmp(image)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, (256, 256))
    image = tf.image.random_crop(image, (224, 224, 3))

    if image_size is not None:
        image = tf.image.resize(image, size=(image_size[0], image_size[1]))
    if tf.shape(image)[-1] == 1:
        image = tf.tile(image, [1,1,3])
    return image

@tf.function
def test_load_image(image_file, image_size=None):
    """ Load the image file.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_bmp(image)
    # image = tf.image.resize(image, (224, 224))
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    if image_size is not None:
        image = tf.image.resize(image, size=(image_size[0], image_size[1]))
    if tf.shape(image)[-1] == 1:
        image = tf.tile(image, [1,1,3])

    return image
