import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
import glob
import gc


from utils import tf_itr
from FCNmodel import build_model

def conv_pred(el):
    t = 10
    idx = np.argsort(el)[::-1]
    return ' '.join(['{}'.format(i) for i in idx[:t]])

def predict(test_relative_path, submission_path, FOLDER):
    model = build_model()
    batch = 100000
    label_num = 3862
    wfn = sorted(glob.glob('weights/*.h5'))[-1]
    model.load_weights(wfn)
    print('loaded weight file: %s' % wfn)
    cnt = 0
    for d in tf_itr(test_relative_path, batch, label_num=label_num, FOLDER=FOLDER):
        cnt += 1
        idx, x1_val, x2_val, _ = d
        ypd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=1, batch_size=32)
        del x1_val, x2_val

        with Pool() as pool:
            out = pool.map(conv_pred, list(ypd))
        df = pd.DataFrame.from_dict({'VideoId': idx, 'Label': out})
        df.to_csv(submission_path + 'subm' + str(cnt) + '.csv', header=True, index=False, columns=['VideoId', 'Label'])
        gc.collect()

    f_subs = glob.glob(os.path.join(submission_path, "subm*.csv"))
    df = pd.concat((pd.read_csv(f) for f in f_subs))
    df.to_csv(os.path.join(submission_path, "output_518030990028.csv"), index=None)


if __name__ == '__main__':
    test_relative_path = ""  # "test"
    submission_path = ''
    FOLDER = 'train_validation'
    predict(test_relative_path, submission_path, FOLDER)
