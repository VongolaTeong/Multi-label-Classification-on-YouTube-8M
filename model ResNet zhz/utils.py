import numpy as np
from multiprocessing import Pool
import glob
import os
import tensorflow as tf
def ap_at_10(data):
    # based on https://github.com/google/youtube-8m/blob/master/average_precision_calculator.py
    predictions, actuals = data
    n = 10
    total_num_positives = None

    if len(predictions) != len(actuals):
        raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be 'None' or a positive integer."
                             " It was '%s'." % n)
    ap = 0.0
    sortidx = np.argsort(predictions)[::-1]

    if total_num_positives is None:
        numpos = np.size(np.where(actuals > 0))
    else:
        numpos = total_num_positives

    if numpos == 0:
        return 0

    if n is not None:  # in label prediction, the num should be 1 or 0
        numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
        r = min(r, n)
    for i in range(r):
        if actuals[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    return ap

# MAP
# restrict on @K is in ap_at_n
def MAP_at_10(pred, actual):
    lst = zip(list(pred), list(actual))

    with Pool() as pool:
        all = pool.map(ap_at_10, lst)

    return np.mean(all)

# data load generator
def tf_itr(tp='test', batch=1024, label_num=3862, FOLDER=""):
    tfiles = sorted(glob.glob(os.path.join(FOLDER, tp, '*tfrecord')))
    print('total files in %s %d' % (tp, len(tfiles)))
    ids, aud, rgb, lbs = [], [], [], []
    for index_i, fn in enumerate(tfiles):
        print("\rLoading files: [{0:50s}] {1:.1f}%".format('#' * int((index_i + 1) / len(tfiles) * 50),
                                                           (index_i + 1) / len(tfiles) * 100), end="", flush=True)
        for example in tf.python_io.tf_record_iterator(fn):
            tf_example = tf.train.Example.FromString(example)
            ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
            rgb.append(np.array(tf_example.features.feature['mean_rgb'].float_list.value))
            aud.append(np.array(tf_example.features.feature['mean_audio'].float_list.value))

            yss = np.array(tf_example.features.feature['labels'].int64_list.value)
            out = np.zeros(label_num).astype(np.int8)
            for y in yss:
                out[y] = 1
            lbs.append(out)
            if len(ids) >= batch:
                yield np.array(ids), np.array(aud), np.array(rgb), np.array(lbs)
                ids, aud, rgb, lbs = [], [], [], []
        if index_i + 1 == len(tfiles):
            yield np.array(ids), np.array(aud), np.array(rgb), np.array(lbs)
            ids, aud, rgb, lbs = [], [], [], []

