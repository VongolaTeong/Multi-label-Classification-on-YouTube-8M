from keras.layers import Input, Dense, BatchNormalization, Dropout, concatenate,Bidirectional, LSTM, \
     MaxPooling2D,MaxPooling1D, Reshape
#from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

# model

embedding_size = 512
lstm_size = 1024
label_size = 3862

def build_model():
    in1 = Input((128,), name='x1')
    x1 = Dense(embedding_size, init='glorot_normal',activation='tanh')(in1)
    x1 = BatchNormalization()(x1)
    x11 = Reshape((1,embedding_size))(x1)
    x2 = Bidirectional(LSTM(lstm_size,return_sequences=True))(x11)
    # x2 = LSTM(lstm_size,activation='relu',return_sequences=True)(x11)
    x2 = BatchNormalization()(x2)
    x3 = Dropout(0.3)(x2)
    #x3 = MaxPooling1D(pool_size=2, padding='same')(x3)
    x4 = Reshape((lstm_size*2,))(x3)

    
    in2 = Input((1024,), name='x2')
    y1 = Dense(embedding_size, init='glorot_normal',activation='tanh')(in2)
    y1 = BatchNormalization()(y1)
    y11 = Reshape((1,embedding_size))(y1)
    y2 = Bidirectional(LSTM(lstm_size,return_sequences=True))(y11)
    #y2 = LSTM(lstm_size,activation='relu',return_sequences=True)(y11)
    y2 = BatchNormalization()(y2)
    y3 = Dropout(0.3)(y2)
    #y3 = MaxPooling1D(pool_size=2, padding='same')(y3)
    y4 = Reshape((lstm_size*2,))(y3)
    
    
    concat = concatenate([x4,y4],axis=1)
    # f1 = Dense(8192, init='glorot_normal',activation='relu')(concat)
    f2 = Dense(4096, init='glorot_normal',activation='tanh')(concat)
    f2 = BatchNormalization()(f2)
    f2 = LeakyReLU()(f2)
    f2 = Dropout(0.2)(f2)
    out = Dense(label_size,activation='sigmoid',name='output')(f2)
    # out = Reshape((-1,3862))(out)
    

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model