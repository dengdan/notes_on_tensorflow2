import tensorflow as tf
from tensorflow import keras
import numpy as np 

def tf_preprocess(img, label, input_size, is_training):
    img = tf.cast(img, tf.float32)
    if is_training:
        img = tf.image.resize_with_crop_or_pad(
            img, input_size + 4, input_size + 4)
        img = tf.image.random_crop(img, [input_size, input_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=63. / 255.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    else:
        img = tf.image.resize_with_crop_or_pad(
                img, input_size, input_size)
    img = tf.image.per_image_standardization(img)
    return img, label
 
def get_dataset(is_training = True, input_size = 32, batch_size = 1):
    # load cifar10 dataset
    cifar10 = keras.datasets.cifar10
    (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
    
    # create generator
    if is_training:
        X = X_train_full
        y = y_train_full
    else:
        X = X_test
        y = y_test
        
    def gen():
        num_examples = X.shape[0]
        for idx in range(num_examples):
            yield X[idx, ...], y[idx]

    # create dataset
    image_shape = X.shape[1:]
    dataset = tf.data.Dataset.from_generator(generator = gen, 
                 output_types = (tf.uint8, tf.int32), 
                 output_shapes = (image_shape, (1,)))

    # preprocessing
    from functools import partial
    pre_process_fn = partial(tf_preprocess, is_training = is_training, input_size = input_size)
    dataset = dataset.map(map_func = pre_process_fn, num_parallel_calls = 12)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size = 100)
        
    dataset = dataset.batch(batch_size = batch_size)
    return dataset
