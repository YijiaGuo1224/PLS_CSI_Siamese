import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics
from load_dataset import load_csi_data
from utils import min_max_norm, gen_exp_test_dataset, pearson_correlation, _compute_eer

import matplotlib
matplotlib.use('TkAgg')
if __name__ == '__main__':
    model_syn = tf.keras.models.load_model('./model_synthetic/cnn_siamese_model_synthetic.h5', compile=False)
    model_syn.compile()
    model_scenario1 = tf.keras.models.load_model('./model_experiment/cnn_siamese_model_experiment_scenario1.h5', compile=False)
    model_scenario1.compile()
    model_scenario2 = tf.keras.models.load_model('./model_experiment/cnn_siamese_model_experiment_scenario2.h5', compile=False)
    model_scenario2.compile()
    model_scenario3 = tf.keras.models.load_model('./model_experiment/cnn_siamese_model_experiment_scenario3.h5', compile=False)
    model_scenario3.compile()
    model_scenario4 = tf.keras.models.load_model('./model_experiment/cnn_siamese_model_experiment_scenario4.h5', compile=False)
    model_scenario4.compile()

    '''Load dataset'''
    Ts = [0.01, 0.1]
    csi, mac, snr = load_csi_data('./ExperimentalTestDataset/scenario1_12cm.csv', Ts)
    abs_csi = np.abs(csi)

    '''MinMax normalisation'''
    norm_csi = min_max_norm(abs_csi)

    '''Reshape dataset'''
    reshape_csi = np.reshape(norm_csi, (norm_csi.shape[0], 1, norm_csi.shape[1], 1))

    '''Generate test dataset'''
    testX, testY = gen_exp_test_dataset(reshape_csi, mac, 1)

    '''Prediction'''
    pred_syn = model_syn.predict([testX[:, :, :, 0], testX[:, :, :, 1]])
    pred_scenario1 = model_scenario1.predict([testX[:, :, :, 0], testX[:, :, :, 1]])
    pred_scenario2 = model_scenario2.predict([testX[:, :, :, 0], testX[:, :, :, 1]])
    pred_scenario3 = model_scenario3.predict([testX[:, :, :, 0], testX[:, :, :, 1]])
    pred_scenario4 = model_scenario4.predict([testX[:, :, :, 0], testX[:, :, :, 1]])

    '''Pearson correlation'''
    pred_corr = pearson_correlation(testX)

    '''ROC plot'''
    scores_syn = pred_syn
    fpr_syn, tpr_syn, thresholds_syn = metrics.roc_curve(testY, scores_syn, pos_label=1)
    auc_syn = metrics.auc(fpr_syn, tpr_syn)
    eer_syn, thre_syn = _compute_eer(fpr_syn, tpr_syn, thresholds_syn)

    scores_scenario1 = pred_scenario1
    fpr_scenario1, tpr_scenario1, thresholds_scenario1 = metrics.roc_curve(testY, scores_scenario1, pos_label=1)
    auc_scenario1 = metrics.auc(fpr_scenario1, tpr_scenario1)
    eer_scenario1, thre_scenario1 = _compute_eer(fpr_scenario1, tpr_scenario1, thresholds_scenario1)

    scores_scenario2 = pred_scenario2
    fpr_scenario2, tpr_scenario2, thresholds_scenario2 = metrics.roc_curve(testY, scores_scenario2, pos_label=1)
    auc_scenario2 = metrics.auc(fpr_scenario2, tpr_scenario2)
    eer_scenario2, thre_scenario2 = _compute_eer(fpr_scenario2, tpr_scenario2, thresholds_scenario2)

    scores_scenario3 = pred_scenario3
    fpr_scenario3, tpr_scenario3, thresholds_scenario3 = metrics.roc_curve(testY, scores_scenario3, pos_label=1)
    auc_scenario3 = metrics.auc(fpr_scenario3, tpr_scenario3)
    eer_scenario3, thre_scenario3 = _compute_eer(fpr_scenario3, tpr_scenario3, thresholds_scenario3)

    scores_scenario4 = pred_scenario4
    fpr_scenario4, tpr_scenario4, thresholds_scenario4 = metrics.roc_curve(testY, scores_scenario4, pos_label=1)
    auc_scenario4 = metrics.auc(fpr_scenario4, tpr_scenario4)
    eer_scenario4, thre_scenario4 = _compute_eer(fpr_scenario4, tpr_scenario4, thresholds_scenario4)

    scores_corr = 1 - pred_corr
    fpr_corr, tpr_corr, thresholds_corr = metrics.roc_curve(testY, scores_corr, pos_label=1)
    auc_corr = metrics.auc(fpr_corr, tpr_corr)
    eer_corr, thre_corr = _compute_eer(fpr_corr, tpr_corr, thresholds_corr)

    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr_syn, tpr_syn, label='Trained by synthetic dataset, AUC = %.4f' % auc_syn)
    plt.plot(fpr_scenario1, tpr_scenario1, label='Trained by experiment dataset (Scenario I), AUC = %.4f' % auc_scenario1)
    plt.plot(fpr_scenario2, tpr_scenario2, label='Trained by experiment dataset (Scenario II), AUC = %.4f' % auc_scenario2)
    plt.plot(fpr_scenario3, tpr_scenario3, label='Trained by experiment dataset (Scenario III), AUC = %.4f' % auc_scenario3)
    plt.plot(fpr_scenario4, tpr_scenario4, label='Trained by experiment dataset (Scenario IV), AUC = %.4f' % auc_scenario4)
    plt.plot(fpr_corr, tpr_corr, label='Correlation-based authentication, AUC = %.4f' % auc_corr)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.grid()
    plt.xticks(size=6)
    plt.yticks(size=6)
    plt.legend(loc=4)
    plt.show()