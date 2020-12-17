
from keras.layers import Input, Dense, BatchNormalization, Dropout, concatenate,Bidirectional, LSTM, \
     MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

# model
def fc_block(x, n=1024, d=0.2):
    x = Dense(n, init='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x

embedding_size = 512
lstm_size = 1024
label_size = 3862

def build_model():
    in1 = Input((128,), name='x1')
    x1 = Dense(embedding_size, init='glorot_normal',activation='tanh')(in1)
    x2 = Bidirectional(LSTM(lstm_size))(x1)
    x3 = Dropout(0.5)(x2)
    x4 = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x3)
    
    in2 = Input((1024,), name='x2')
    y1 = Dense(embedding_size, init='glorot_normal',activation='tanh')(in2)
    y2 = Bidirectional(LSTM(lstm_size))(y1)
    y3 = Dropout(0.5)(y2)
    y4 = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(y3)
    
    concat = concatenate([x4,y4])
    f1 = Dense(8192, init='glorot_normal',activation='Relu')(concat)
    f2 = Dense(4096, init='glorot_normal',activation='tanh')(f1)
    out = Dense(label_size, init='glorot_normal',activation='sigmoid')(f2)
    

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model