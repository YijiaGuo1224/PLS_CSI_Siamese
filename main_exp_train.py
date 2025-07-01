import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from deep_learning_models import cnn_based_siamese_model
from load_dataset import load_exp_file
from utils import min_max_norm, gen_exp_train_dataset, shuffle, loss

import matplotlib
matplotlib.use('TkAgg')
if __name__ == '__main__':
    '''Load dataset'''
    path = ['./ExperimentalTrainingDataset/scenario1_training.csv']
    # path = ['./ExperimentalTrainingDataset/scenario2_training1.csv', './ExperimentalTrainingDataset/scenario2_training2.csv']
    # path = ['./ExperimentalTrainingDataset/scenario3_training1.csv', './ExperimentalTrainingDataset/scenario3_training2.csv']
    # path = ['./ExperimentalTrainingDataset/scenario4_training.csv']
    abs_csi, num_pkt, sum_pkt = load_exp_file(path)
    
    '''MinMax normalisation'''
    norm_csi = min_max_norm(abs_csi)

    '''Reshape dataset'''
    reshape_csi = np.reshape(norm_csi, (norm_csi.shape[0], 1, norm_csi.shape[1], 1))

    '''Generate training dataset and shuffle data'''
    trainX, trainY = gen_exp_train_dataset(reshape_csi, num_pkt, sum_pkt)
    shuff_trainX, shuff_trainY = shuffle(trainX, trainY)

    '''Define the neural network'''
    model = cnn_based_siamese_model(shuff_trainX)
    early_stop = EarlyStopping('val_loss', min_delta=0, patience=20)
    reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
    callbacks = [early_stop, reduce_lr]

    '''Train the neural network'''
    model.compile(loss=loss(margin=1), optimizer="RMSprop", metrics=["accuracy"])

    history = model.fit([shuff_trainX[:, :, :, 0], shuff_trainX[:, :, :, 1]],
                        shuff_trainY,
                        epochs=5000,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=1,
                        batch_size=32,
                        callbacks=callbacks)

    tf.keras.models.save_model(model, './model_experiment/cnn_siamese_model_experiment_scenario1.h5')