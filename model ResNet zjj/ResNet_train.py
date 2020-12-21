# MAP@10 0.76984

import os
from utils import tf_itr, MAP_at_10
from ResNet_FCNmodel import build_model

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
    train_relative_path = '../test_train'
    val_relative_path = '../test_validation'
    FOLDER = ''
    train(train_relative_path, val_relative_path, FOLDER)
