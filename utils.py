import numpy as np
import tensorflow as tf

def convert2complex(data_in):
    num_sample = data_in.shape[0]
    signal_length = round(data_in.shape[1] / 2)
    data_out = np.empty([num_sample, signal_length], dtype=complex)
    data_out = data_in[:, :signal_length] + 1j * data_in[:, signal_length:]
    return data_out

def min_max_norm(data_in):
    data_min = data_in.min()
    data_max = data_in.max()
    data_out = (data_in - data_min) / (data_max - data_min)
    return data_out

def gen_syn_dataset(data_in1, data_in2, data_in3):
    trainX1 = np.concatenate((data_in1, data_in2), axis=-1)
    trainX2 = np.concatenate((data_in1, data_in3), axis=-1)
    trainY1 = np.zeros((len(trainX1)), dtype=float)
    trainY2 = np.ones((len(trainX2)), dtype=float)

    trainX = np.concatenate((trainX1, trainX2), axis=0)
    trainY = np.concatenate((trainY1, trainY2), axis=0)
    return trainX, trainY

def gen_exp_train_dataset_path(data_in, look_back):
    data_out = []
    for i in range(len(data_in)-look_back):
        a = np.concatenate((data_in[i, :], data_in[i + look_back, :]), axis=-1)
        data_out.append(a)
    data_out = np.array(data_out)
    return data_out

def gen_exp_train_dataset(data_in, num_pkt, sum_pkt):
    for i in range(len(num_pkt)):
        pkt_idx = np.arange(sum_pkt[i], sum_pkt[i + 1])
        trainX1_path = gen_exp_train_dataset_path(data_in[pkt_idx], 1)
        trainX2_path = gen_exp_train_dataset_path(data_in[pkt_idx], 100)
        trainY1_path = np.zeros((len(trainX1_path)), dtype=float)
        trainY2_path = np.ones((len(trainX2_path)), dtype=float)
        if i == 0:
            trainX1 = trainX1_path
            trainX2 = trainX2_path
            trainY1 = trainY1_path
            trainY2 = trainY2_path
        else:
            trainX1 = np.append(trainX1, trainX1_path, axis=0)
            trainX2 = np.append(trainX2, trainX2_path, axis=0)
            trainY1 = np.append(trainY1, trainY1_path, axis=0)
            trainY2 = np.append(trainY2, trainY2_path, axis=0)

    num_training_pkt = min(len(trainX1), len(trainX2))
    trainX = np.concatenate((trainX1[:num_training_pkt], trainX2[:num_training_pkt]), axis=0)
    trainY = np.concatenate((trainY1[:num_training_pkt], trainY2[:num_training_pkt]), axis=0)
    return trainX, trainY

def gen_exp_test_dataset(data_in, mac, look_back):
    testX = []
    testY = []
    for i in range(len(data_in)-look_back):
        a = np.concatenate((data_in[i, :], data_in[i + look_back, :]), axis=-1)
        testX.append(a)
        if mac[i] == mac[i + look_back]:
            b = float(0)
        else:
            b = float(1)
        testY.append(b)
    testX = np.array(testX)
    testY = np.array(testY)
    return testX, testY

def shuffle(data_in, label_in):
    index = np.arange(len(data_in))
    np.random.shuffle(index)
    data_out = data_in[index]
    label_out = label_in[index]
    return data_out, label_out

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return contrastive_loss

def pearson_correlation(data_in):
    num_sample = data_in.shape[0]
    corr = np.empty(num_sample, dtype=float)
    for i in range(num_sample):
        mean_data_in1 = np.mean(data_in[i, 0, :, 0])
        mean_data_in2 = np.mean(data_in[i, 0, :, 1])
        a = np.vdot((data_in[i, 0, :, 0]-mean_data_in1), (data_in[i, 0, :, 1]-mean_data_in2))
        b = np.linalg.norm(data_in[i, 0, :, 0]-mean_data_in1)*np.linalg.norm(data_in[i, 0, :, 1]-mean_data_in2)
        corr[i] = a/b
    return corr

def _compute_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]