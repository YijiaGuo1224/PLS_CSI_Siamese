import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from deep_learning_models import cnn_based_siamese_model
from load_dataset import load_sim_file
from utils import min_max_norm, gen_syn_dataset, shuffle, loss

if __name__ == '__main__':
    '''Load dataset'''
    h_ba_t0_path = './SyntheticTrainingDataset/Train_h_ba_t0.h5'
    h_ba_t1_path = './SyntheticTrainingDataset/Train_h_ba_t1.h5'
    h_ma_t1_path = './SyntheticTrainingDataset/Train_h_ma_t1.h5'

    h_ba_t0 = load_sim_file(h_ba_t0_path, 'dataset')
    h_ba_t1 = load_sim_file(h_ba_t1_path, 'dataset')
    h_ma_t1 = load_sim_file(h_ma_t1_path, 'dataset')
    
    '''MinMax normalisation'''
    norm_h_ba_t0 = min_max_norm(h_ba_t0)
    norm_h_ba_t1 = min_max_norm(h_ba_t1)
    norm_h_ma_t1 = min_max_norm(h_ma_t1)

    '''Reshape dataset'''
    reshape_h_ba_t0 = np.reshape(norm_h_ba_t0, (norm_h_ba_t0.shape[0], 1, norm_h_ba_t0.shape[1], 1))
    reshape_h_ba_t1 = np.reshape(norm_h_ba_t1, (norm_h_ba_t1.shape[0], 1, norm_h_ba_t1.shape[1], 1))
    reshape_h_ma_t1 = np.reshape(norm_h_ma_t1, (norm_h_ma_t1.shape[0], 1, norm_h_ma_t1.shape[1], 1))

    '''Generate training dataset and shuffle data'''
    trainX, trainY = gen_syn_dataset(reshape_h_ba_t0, reshape_h_ba_t1, reshape_h_ma_t1)
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

    tf.keras.models.save_model(model, './model_synthetic/cnn_siamese_model_synthetic.h5')