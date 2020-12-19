from keras.layers import Input, Dense, BatchNormalization, Dropout, concatenate,Bidirectional, LSTM, \
     MaxPooling1D, Reshape, Activation
#from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU

from keras.models import Model

# model

embedding_size = 1024
lstm_size = 1024
label_size = 3862

def build_model():
    in1 = Input((128,), name='x1')
    x1 = Dense(embedding_size, init='glorot_normal',activation='tanh')(in1)
    #x1 = LeakyReLU()(x1)
    #x1 = Activation('tanh')(x1)
    x1 = BatchNormalization()(x1)
    x11 = Reshape((1,embedding_size))(x1)
    x2 = Bidirectional(LSTM(lstm_size,return_sequences=True))(x11)

    x2 = BatchNormalization()(x2)
    x3 = Dropout(0.3)(x2)
    #x3 = MaxPooling(pool_size=2, padding='same')(x3)
    x4 = Reshape((lstm_size*2,))(x3)

    
    in2 = Input((1024,), name='x2')
    y1 = Dense(embedding_size, init='glorot_normal',activation='tanh')(in2)
    #y1 = LeakyReLU()(y1)
    #y1 = Activation('tanh')(y1)
    y1 = BatchNormalization()(y1)
    y11 = Reshape((1,embedding_size))(y1)
    y2 = Bidirectional(LSTM(lstm_size,return_sequences=True))(y11)

    y2 = BatchNormalization()(y2)
    y3 = Dropout(0.3)(y2)
    #y3 = MaxPooling(pool_size=2, padding='same')(y3)
    y4 = Reshape((lstm_size*2,))(y3)
    
    
    concat = concatenate([x4,y4],axis=1)
    f1 = Dense(4096, activation='tanh')(concat)
    #f1 = BatchNormalization()(f1)
    #f2 = Dense(4000, activation='tanh')(f1)
    f2 = BatchNormalization()(f1)
    f2 = LeakyReLU()(f2)
    f2 = Dropout(0.2)(f2)
    out = Dense(label_size,activation='sigmoid',name='output')(f2)
    

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model