from keras.layers import Input, Dense, concatenate
from keras.models import Model
import tensorflow as tf

# model
def nn_block(x, n=1024, d=0.2):
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
