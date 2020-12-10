
from keras.layers import Input, Dense, BatchNormalization, Dropout, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

# model
def fc_block(x, n=1024, d=0.2):
    x = Dense(n, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x

def build_model():
    in1 = Input((128,), name='x1')
    x1 = fc_block(in1)
    x1 = fc_block(x1)
    x1 = Dense(1024 , kernel_initializer='glorot_normal')(x1)
    x3 = fc_block(in1)
    y1 = concatenate([x1, x3], axis=1)
    out1 = LeakyReLU()(y1)

    in2 = Input((1024,), name='x2')
    x2 = fc_block(in2)
    x2 = fc_block(x2)
    x2 = Dense(1024 , kernel_initializer='glorot_normal')(x2)
    x4 = fc_block(in2)
    y2 = concatenate([x2, x4], axis=1)
    out2 = LeakyReLU()(y2)
    
    
    x = concatenate([out1, out2], axis=1)
    x = fc_block(x)
    out3 = Dense(1024 , kernel_initializer='glorot_normal')(x)
    
    z = concatenate([out1, out2], axis=1)
    z1 = concatenate([out3, z], axis=1)
    z1 = LeakyReLU()(z1)
    z1 = fc_block(z1)
    out = Dense(3862, activation='sigmoid', name='output')(z1)

    model = Model(inputs=[in1, in2], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model
