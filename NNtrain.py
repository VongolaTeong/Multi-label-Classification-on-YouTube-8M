# MAP@10 0.76984

import os
from utils import tf_itr, MAP_at_10
from NNmodel import build_model

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

def train(train_relative_path, val_relative_path, FOLDER):
    if not os.path.exists('weights'): os.mkdir('weights')
    batch = 10*1024
    n_itr = 10
    n_eph = 100
    label_num = 3862
    _, x1_val, x2_val, y_val = next(tf_itr(val_relative_path, 10000, label_num=label_num, FOLDER=FOLDER))
    model = build_model()
    cnt = 0
    for e in range(n_eph):
        for d in tf_itr(train_relative_path, batch, label_num=label_num, FOLDER=FOLDER):
            _, x1_trn, x2_trn, y_trn = d
            model.train_on_batch({'x1': x1_trn, 'x2': x2_trn}, {'output': y_trn})
            cnt += 1
            if cnt % n_itr == 0:
                y_prd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=False, batch_size=100)
                g = MAP_at_10(y_prd, y_val)
                print('val GAP %0.5f; epoch: %d; iters: %d' % (g, e, cnt))
                model.save_weights('weights/%0.5f_%d_%d.h5' % (g, e, cnt))



if __name__ == '__main__':
    train_relative_path = 'train'
    val_relative_path = 'validation'
    FOLDER = ''
    train(train_relative_path, val_relative_path, FOLDER)
