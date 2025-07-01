from tensorflow import keras
from tensorflow.keras import layers
from utils import euclidean_distance

def cnn_based_siamese_model(data_in):
    input = layers.Input(shape=(data_in.shape[1], data_in.shape[2], 1))
    x = layers.Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), activation='relu', padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 1))(x)
    x = layers.Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 1))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    embedding_network = keras.Model(input, x)

    input_1 = layers.Input(shape=(data_in.shape[1], data_in.shape[2], 1))
    input_2 = layers.Input(shape=(data_in.shape[1], data_in.shape[2], 1))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    norm_layer = layers.BatchNormalization()(merge_layer)
    output = layers.Dense(1, activation="sigmoid")(norm_layer)

    cnn_based_siamese = keras.Model(inputs=[input_1, input_2], outputs=output)
    cnn_based_siamese.summary()
    return cnn_based_siamese