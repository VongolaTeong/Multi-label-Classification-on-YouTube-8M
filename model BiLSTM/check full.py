import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
import glob
import gc


from utils import tf_itr, MAP_at_10
from bilstm import build_model

def conv_pred(el):
    t = 10
    idx = np.argsort(el)[::-1]
    return ' '.join(['{}'.format(i) for i in idx[:t]])

gsum=0
gcount=0

def predict(test_relative_path, submission_path, FOLDER1, FOLDER2):
    model = build_model()
    batch = 100000
    label_num = 3862
    wfn = sorted(glob.glob('weights/*.h5'))[-1]
    model.load_weights(wfn)
    print('loaded weight file: %s' % wfn)
    cnt = 0
    global gsum,gcount
    for d in tf_itr(test_relative_path, batch, label_num=label_num, FOLDER=FOLDER1):
        cnt += 1
        idx, x1_val, x2_val, yval = d
        ypd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=1, batch_size=32)
        
        gsum += MAP_at_10(ypd, yval)
        gcount += 1
        del x1_val, x2_val

        with Pool() as pool:
            out = pool.map(conv_pred, list(ypd))
        df = pd.DataFrame.from_dict({'VideoId': idx, 'Label': out})
        df.to_csv(submission_path + 'subt' + str(cnt) + '.csv', header=True, index=False, columns=['VideoId', 'Label'])
        gc.collect()
        
    for d in tf_itr(test_relative_path, batch, label_num=label_num, FOLDER=FOLDER2):
        cnt += 1
        idx, x1_val, x2_val, yval = d
        ypd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=1, batch_size=32)
        gsum += MAP_at_10(ypd, yval)
        gcount += 1
        del x1_val, x2_val

        with Pool() as pool:
            out = pool.map(conv_pred, list(ypd))
        df = pd.DataFrame.from_dict({'VideoId': idx, 'Label': out})
        df.to_csv(submission_path + 'subv' + str(cnt) + '.csv', header=True, index=False, columns=['VideoId', 'Label'])
        gc.collect()

    f_subs = glob.glob(os.path.join(submission_path, "sub*.csv"))
    df = pd.concat((pd.read_csv(f) for f in f_subs))
    df.to_csv(os.path.join(submission_path, "output_518030990028.csv"), index=None)


if __name__ == '__main__':
    test_relative_path = ""  # "test"
    submission_path = ''
    FOLDER1 = '../../../train'
    FOLDER2 = '../../../validation'
    predict(test_relative_path, submission_path, FOLDER1, FOLDER2)
    with open("MAP.txt","w") as f:
        f.write(str(gsum/gcount))
