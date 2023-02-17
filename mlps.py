import pickle
import numpy as np
from datetime import date
from sklearn.neural_network import MLPClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm


class MLP:
    def __init__(self, model):
        self.clf = model

    def predict(self, array, as_probabilities=False, no_data=None):
        if isinstance(self.clf, MLPClassifier):
            classifier = self.clf
        else:
            classifier = convert_to_sk_mlp(self.clf)
        if no_data is None:
            if array.ndim == 3:
                if as_probabilities:
                    output = np.zeros((array.shape[0], array.shape[1], len(classifier.classes_)))
                else:
                    output = np.zeros((array.shape[0], array.shape[1]))
                for i in tqdm(range(array.shape[0])):
                    if as_probabilities:
                        predictions = classifier.predict_proba(array[i, ...].reshape(-1, array.shape[2]))
                    else:
                        predictions = classifier.predict(array[i, ...].reshape(-1, array.shape[2]))
                    output[i, :] = predictions
                if as_probabilities:
                    return output
                else:
                    return output.astype(int)
            else:  # if input is in Y_validate form i.e. shape: n_samples x n_channels
                if as_probabilities:
                    return classifier.predict_proba(array)
                else:
                    return classifier.predict(array).astype(int)
        else:
            last_axis = array.ndim - 1
            if as_probabilities:
                output = np.zeros((array.shape[:-1] + (len(classifier.classes_),)), dtype=np.float32)
                if last_axis == 2 and array.shape[0] * array.shape[0] > 1000 * 1000:
                    for i in tqdm(range(array.shape[0])):
                        idxs = np.where(~(array[i] == no_data).all(axis=last_axis - 1))
                        row = array[i][idxs]
                        if len(row) > 0:
                            output[i][idxs] = classifier.predict_proba(row)
                else:
                    idxs = np.where(~(array == no_data).all(axis=last_axis))
                    output[idxs] = classifier.predict_proba(array[idxs])
                return output
            else:
                output = np.zeros(array[..., 0].shape)
                idxs = np.where(~(array == no_data).all(axis=last_axis))
                output[idxs] = classifier.predict(array[idxs])
                return output.astype(int)

    def train(self, ip_op_data, save_directory="models/MLPs", iteration_number=""):
        X_train, X_validate, Y_train, Y_validate = ip_op_data
        if isinstance(self.clf, MLPClassifier):
            self.clf.fit(X_train, Y_train)
        else:
            with open(save_directory + "/" + "mlp" + str(iteration_number) + ".json",
                      "w") as json_file:
                json_file.write(self.clf.to_json())
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = ModelCheckpoint(
                save_directory + '/' + "mlp" + str(iteration_number) + ".h5",
                monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=False)
            history = self.clf.fit(X_train, Y_train, epochs=600, validation_data=(X_validate, Y_validate),
                                   callbacks=[es, mc], batch_size=128)

    def dump(self, path='mlp.pkl'):
        with open(path, 'wb') as file:
            output = pickle.dump(self.clf, file)
        return output  # pickle.dump returns None so it was as if there was no return statement (useful for testing)


def from_pkl(path):
    if isinstance(path, MLPClassifier):
        raise TypeError(
            "mlps.from_pkl() takes a string as argument but MLPClassifier object was given. Try mlps.from_model()")
    if not isinstance(path, str):
        raise TypeError(
            "mlps.from_pkl() takes a string as argument but an object of type %s was given. Try mlps.from_model()" % type(
                path)
        )
    with open(path, 'rb') as file:
        model = pickle.load(file)

    return MLP(model)


def from_model(model):
    if isinstance(model, str):
        raise TypeError(
            "mlps.from_model() takes an MLPClassifier/keras Sequential object as argument but a str was given.")
    return MLP(model)


def from_file(model_json, weights_path):
    with open(model_json, 'r') as json_file:
        model = models.model_from_json(json_file.read())
    model.load_weights(weights_path)
    return MLP(model)


def resume_from_file(model_json, weights_path):
    return from_model(tf.keras.models.load_model(weights_path))


def make_model(numb_input_channels=3, numb_outputs=12):
    tf.random.set_seed(1)
    model = models.Sequential()
    model.add(layers.Dense(units=32, activation='relu', input_shape=(numb_input_channels,)))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(numb_outputs, activation='softmax'))

    model.compile(
        optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


@ignore_warnings(category=ConvergenceWarning)
def convert_to_sk_mlp(model):
    numb_inputs = model.input_shape[-1]
    hidden_layer_sizes = tuple([layer.output_shape[-1] for layer in model.layers[:-1]])
    activation_functions = [layer.get_config()['activation'] for layer in model.layers]
    numb_outputs = model.layers[-1].output_shape[-1]
    sk_mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                           activation=activation_functions[0],
                           max_iter=1,
                           random_state=1)
    sk_mlp.fit(np.random.random((numb_outputs, numb_inputs)), np.arange(numb_outputs) + 1)
    weights = model.get_weights()
    coeff_list = [w for i, w in enumerate(weights) if i % 2 == 0]
    intercepts_list = [w for i, w in enumerate(weights) if i % 2 == 1]
    sk_mlp.coefs_ = coeff_list
    sk_mlp.intercepts_ = intercepts_list
    return sk_mlp
