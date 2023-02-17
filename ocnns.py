from datetime import date
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, activations, Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
import numpy as np


class OCNN:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, X_validate, Y_train, Y_validate, save_directory="models/OCNNs", iteration_number=""):
        with open(save_directory + "/" + "ocnn" + str(iteration_number) + ".json",
                  "w") as json_file:
            json_file.write(self.model.to_json())
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(
            save_directory + '/' + "ocnn" + str(iteration_number)+'.h5',
            monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=False)
        history = self.model.fit(X_train, Y_train, epochs=600, validation_data=(X_validate, Y_validate),
                                 callbacks=[es, mc], batch_size=4)

    def predict(self, mlp_output, as_probabilities=False):
        def chunk_predict(model, mlp_op, probabilities, op=[]):
            if len(mlp_output) <= 1028:
                try:
                    if probabilities:
                        op.append(model.predict(np.concatenate(mlp_op)))
                    else:
                        op.append(model.predict(np.concatenate(mlp_op)).argmax(axis=1))
                except MemoryError as e:
                    print(e)
                    half_idx = int(len(mlp_op) / 2)
                    op = chunk_predict(model, mlp_op[:half_idx], as_probabilities, op)
                    op = chunk_predict(model, mlp_op[half_idx:], as_probabilities, op)
            else:
                half_idx = int(len(mlp_op) / 2)
                op = chunk_predict(model, mlp_op[:half_idx], as_probabilities, op)
                op = chunk_predict(model, mlp_op[half_idx:], as_probabilities, op)
            return op

        print('Using CNN to make predictions')
        Y = chunk_predict(self.model, mlp_output, as_probabilities)
        Y = np.concatenate(Y)
        print('Finished predictions')
        return Y


def from_file(model_json, weights_path):
    with open(model_json, 'r') as json_file:
        model = models.model_from_json(json_file.read())
    model.load_weights(weights_path)
    return OCNN(model)


def resume_from_file(model_json, weights_path):
    return from_model(tf.keras.models.load_model(weights_path))


def from_model(model):
    return OCNN(model)


def make_model(window_size=96, numb_input_channels=12, numb_outputs=8):
    # trained_model = ResNet152V2(include_top=False, input_shape=(window_size, window_size, numb_input_channels))
    # inputs = Input(shape=(window_size, window_size, 3))
    # x = trained_model(inputs, training=False)
    # x = layers.Flatten()(x)
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dense(512, activation='relu')(x)
    # outputs = layers.Dense(numb_outputs, activation='softmax')(x)
    # model = Model(inputs, outputs)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(window_size, window_size, numb_input_channels)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(32, kernel_size=3))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.MaxPooling2D(pool_size=2))
    # model.add(layers.Conv2D(32, kernel_size=3))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation(activations.relu))
    # model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(numb_outputs, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001/100),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  )
    return model
