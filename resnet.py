import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf
import tensorflow.keras as keras
layers = keras.layers

def block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    preact = layers.BatchNormalization()(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x

def stack_fn(x):
    x = stack2(x, 64, 3, name='conv2')
    x = stack2(x, 128, 4, name='conv3')
    x = stack2(x, 256, 6, name='conv4')
    x = stack2(x, 512, 3, stride1=1, name='conv5')
    return x

def ResNet(input_tensor, preact, use_bias, classes=10):
    x = input_tensor
    x = layers.Conv2D(64, 3, strides=1, use_bias=use_bias, name='conv1_1')(x)
    x = layers.Conv2D(64, 3, strides=1, use_bias=use_bias, name='conv1_2')(x)
    if preact is False:
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='post_relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='probs')(x)
    return x

input_size = 32
input_tensor = layers.Input(shape = (input_size, input_size, 3), dtype = tf.float32)
output_tensor = ResNet(input_tensor, preact = True, use_bias = True)
model = keras.Model(inputs = input_tensor, outputs = output_tensor)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

import cifar_input
train_batch_size = 100
eval_batch_size = 100
ds_train = cifar_input.get_dataset(is_training = True, input_size = input_size, batch_size = train_batch_size)
ds_eval = cifar_input.get_dataset(is_training = False, input_size = input_size, batch_size = eval_batch_size)

import util
run_logdir = util.io.get_absolute_path("~/temp/no-use/keras/resent")
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(run_logdir)

model.fit(x = ds_train, epochs=300, validation_data = ds_eval, callbacks=[checkpoint_cb, tensorboard_cb])