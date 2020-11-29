from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import tensorflow as tf

# model
def fc_block(x, n=1024, d=0.2):
    x = Dense(n, init='glorot_normal')(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1)(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x

def build_model():
    in11 = Input(shape = (128,), name='x1')
    in1 = Input(shape = (None, None, 128))
    x1 = fc_block(in1)
    in22 = Input((1024,), name='x2')
    in2 = Input(shape = (None, None, 1024))
    x2 = fc_block(in2)
    x = concatenate([x1, x2], axis=1)
    x = fc_block(x)
    out = Dense(3862, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

    
