from keras.layers import Input, Dense, BatchNormalization, Dropout, concatenate, merge, Average
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

#fc block
def fc_block(x, n=1024, d=0.2):
    x = Dense(n, init='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x

#identity mapping function which used to skip blocks
def identity_map(input_x, n=1024, d=0.2):
    x = fc_block(input_x, n, d)
    x = Dense(int(input_x.shape[1]))(x)
    x = Average()([x, input_x])
    x = LeakyReLU()(x)
    return x

#model
def build_model():
    #fc block 1
    in1 = Input((128,), name='x1')
    x1 = fc_block(in1)

    #identity map x1
    x1 = identity_map(x1)

    #fc block 2
    in2 = Input((1024,), name='x2')
    x2 = fc_block(in2)

    #identity map x2
    x2 = identity_map(x2)

    #merge x1 and x2
    x = concatenate([x1, x2], axis=1)

    #identity map x
    x = identity_map(x)

    #output
    out = Dense(3862, activation='sigmoid', name='output')(x)
    
    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

