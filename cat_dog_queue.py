import tensorflow as tf
import numpy as np
import os
import ntpath


def get_image_paths():
    cat_files = [
        'data/dogs_and_cats/cats/' + f
        for 
        f
        in
        os.listdir('data/dogs_and_cats/cats')
    ]

    dog_files = [
        'data/dogs_and_cats/dogs/' + f
        for 
        f
        in
        os.listdir('data/dogs_and_cats/dogs')
    ]
    return cat_files + dog_files


def clean_image_data(image, pixel_depth, height, width):
    norm = 2 * ((tf.cast(image, tf.float32) - (pixel_depth / 2)) / pixel_depth)
    resized = tf.image.resize_images([norm], height, width)
    return resized


def cat_or_dog(filename):
    name = ntpath.basename(filename)
    if b'dog' in name:
        return np.array(1, dtype=np.int32)
    else:
        return np.array(0, dtype=np.int32)


def get_queue(pixel_depth, height, width, channels, batch_size):
    filename_queue = tf.train.string_input_producer(get_image_paths())
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    label = tf.py_func(cat_or_dog, [key], [tf.int32])[0]
    label = tf.reshape(label, [1])
    image = tf.image.decode_jpeg(value, channels=channels)
    cleaned = clean_image_data(image, pixel_depth, height, width)
    min_after_dequeue = 2000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [cleaned, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    image_squeezed = tf.squeeze(image_batch)
    return image_squeezed, label_batch