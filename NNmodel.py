from keras.layers import Input, Dense, concatenate
from keras.models import Model
import tensorflow as tf
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 100

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# model
def nn_block(x):
    W1 = tf.Variable(tf.random_normal([1152,2048], stddev=0.01))
    L1 = tf.nn.relu(tf.matmul(x, W1))

    W2 = tf.Variable(tf.random_normal([2048,4096], stddev=0.01))
    L2 = tf.nn.relu(tf.matmul(L1, W2))

    W3 = tf.Variable(tf.random_normal([4096, 3862], stddev=0.01))
    output = tf.matmul(L2, W3)
    output = tf.nn.softmax(output)
    return x

def build_model():
    in1 = Input((128,), name='x1')
    in2 = Input((1024,), name='x2')

    x = concatenate([in1, in2], axis=1)
    x = nn_block(x)
    out = Dense(3862, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model
