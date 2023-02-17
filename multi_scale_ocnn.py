from tensorflow.keras import models, layers, activations, optimizers, losses, initializers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np
import os


class HalfInputs(layers.Layer):
    def __init__(self):
        super(HalfInputs, self).__init__()

    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, (int(inputs.shape[1] / 2), int(inputs.shape[2] / 2)))


class ZoneInputs(layers.Layer):
    def __init__(self):
        super(ZoneInputs, self).__init__()

    def call(self, inputs, **kwargs):
        return inputs[:, int(inputs.shape[1] / 4):int(inputs.shape[1] / 4) + int(inputs.shape[1] / 2),
               int(inputs.shape[2] / 4):int(inputs.shape[2] / 4) + int(inputs.shape[2] / 2), :]


class CNN(layers.Layer):
    def __init__(self, output_shape=None):
        super(CNN, self).__init__()
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.relu = activations.relu
        self.conv0 = layers.Conv2D(filters=32, kernel_size=3, padding="same")
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, padding="same")
        self.conv2 = layers.Conv2D(filters=32, kernel_size=3, padding="same")
        self.desired_output_shape = output_shape

    def call(self, inputs, **kwargs):
        x = self.conv0(inputs)
        x = self.max_pool(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        return self.relu(x) if self.desired_output_shape is None else tf.image.resize(self.relu(x),
                                                                                      self.desired_output_shape)


class WeightedCombine(layers.Layer):
    def __init__(self):
        super(WeightedCombine, self).__init__()
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(name="w", shape=(input_shape[-1],),
                                 initializer=initializers.RandomNormal(mean=1 / 3, stddev=0.05, seed=None))

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(tf.multiply(inputs, self.w[None, None, None, None, :]), -1)


class MultiScaleModel(models.Model):
    def __init__(self, numb_outputs):
        super(MultiScaleModel, self).__init__()

        self.half_inputs = HalfInputs()
        self.cnn = None
        self.cnn1 = None
        self.cnn2 = None
        self.wc = WeightedCombine()
        self.dense = layers.Dense(32, activation=activations.relu)
        self.op_layer = layers.Dense(numb_outputs, activation=activations.softmax)

    def build(self, input_shape):
        self.cnn = CNN()
        cnn_output_shape = round((round((round((input_shape[1]) / 2)) / 2)) / 2)
        self.cnn1 = CNN((cnn_output_shape, cnn_output_shape))
        self.cnn2 = CNN((cnn_output_shape, cnn_output_shape))

    def call(self, inputs, training=None, mask=None):
        # inputs shape (None, WS,WS,channels)
        inputs_1 = self.half_inputs(inputs)
        inputs_2 = self.half_inputs(inputs_1)

        features = self.cnn(inputs)
        features_1 = self.cnn1(inputs_1)
        features_2 = self.cnn2(inputs_2)

        features = tf.stack([features, features_1, features_2], -1)
        features = self.wc(features)

        x = layers.Flatten()(features)
        x = self.dense(x)
        return self.op_layer(x)


class BenchmarkModel(models.Model):
    def __init__(self, numb_outputs):
        super(BenchmarkModel, self).__init__()

        self.half_inputs = HalfInputs()
        self.cnn = CNN()
        self.dense = layers.Dense(32, activation=activations.relu)
        self.op_layer = layers.Dense(numb_outputs, activation=activations.softmax)

    def call(self, inputs, training=None, mask=None):
        features = self.cnn(inputs)
        x = layers.Flatten()(features)
        x = self.dense(x)
        return self.op_layer(x)


class MultiZoneModel(models.Model):
    def __init__(self, numb_outputs):
        super(MultiZoneModel, self).__init__()

        self.zone_inputs = ZoneInputs()
        self.cnn = CNN()
        # self.cnn1 = CNN((22, 22))
        # self.cnn2 = CNN((22, 22))
        # self.wc = WeightedCombine()
        self.dense = layers.Dense(32, activation=activations.relu)
        self.op_layer = layers.Dense(numb_outputs, activation=activations.softmax)

    def call(self, inputs, training=None, mask=None):
        # inputs shape (None, WS,WS,channels)
        # inputs_1 = self.zone_inputs(inputs)
        # inputs_2 = self.zone_inputs(inputs_1)

        features = self.cnn(inputs)
        # features_1 = self.cnn1(inputs_1)
        # features_2 = self.cnn2(inputs_2)

        # features = tf.stack([features, features_1, features_2], -1)
        # features = self.wc(features)

        x = layers.Flatten()(features)
        x = self.dense(x)
        return self.op_layer(x)


def train_multiscale_model(train_data_tuple, validation_data_tuple, save_directory, name):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    x_train, y_train = train_data_tuple
    x_validate, y_validate = validation_data_tuple

    # Random shuffle
    np.random.seed(1)
    train_sample = np.random.choice(np.arange(x_train.shape[0]), x_train.shape[0], replace=False)
    validate_sample = np.random.choice(np.arange(x_validate.shape[0]), x_validate.shape[0], replace=False)
    
    x_train = x_train[train_sample, ..., :3]
    y_train = y_train[train_sample]
    
    x_validate = x_validate[validate_sample, ..., :3]
    y_validate = y_validate[validate_sample]

    # Check for nans/infs
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(y_train))
    assert not np.any(np.isnan(x_validate))
    assert not np.any(np.isnan(y_validate))

    print("\n##############################")
    print(f"GENERATING MULTI-SCALE MODEL")
    print("##############################\n")

    ms_model = MultiScaleModel(y_train.shape[-1])
    ms_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                     loss=losses.categorical_crossentropy,
                     metrics=['accuracy'],
                     )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(os.path.join(save_directory, name + ".h5"),
                         monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    
    ms_model.fit(x_train, y_train, epochs=100,
                 validation_data=(x_validate, y_validate),
                 callbacks=[es, mc], batch_size=64)

    return ms_model


def train_ocnn_model(train_data_tuple, validation_data_tuple, save_directory, name):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    x_train, y_train = train_data_tuple
    x_validate, y_validate = validation_data_tuple

    # Random shuffle
    np.random.seed(1)
    train_sample = np.random.choice(np.arange(x_train.shape[0]), x_train.shape[0], replace=False)
    validate_sample = np.random.choice(np.arange(x_validate.shape[0]), x_validate.shape[0], replace=False)
    
    x_train = x_train[train_sample, ..., :3]
    y_train = y_train[train_sample]
    
    x_validate = x_validate[validate_sample, ..., :3]
    y_validate = y_validate[validate_sample]

    # Check for nans/infs
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(y_train))
    assert not np.any(np.isnan(x_validate))
    assert not np.any(np.isnan(y_validate))

    print("\n#################################")
    print(f"GENERATING OCNN BENCHMARK MODEL")
    print("#################################\n")

    benchmark_model = BenchmarkModel(y_train.shape[-1])
    benchmark_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                            loss=losses.categorical_crossentropy,
                            metrics=['accuracy'],
                            )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(os.path.join(save_directory, name + "_benchmark.h5"),
                         monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    
    benchmark_model.fit(x_train, y_train, epochs=100,
                        validation_data=(x_validate, y_validate),
                        callbacks=[es, mc], batch_size=64)

    return benchmark_model

def train_adaptive_multiscale_model(train_data_tuple, validation_data_tuple, save_directory, name):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    x_train, y_train = train_data_tuple
    x_validate, y_validate = validation_data_tuple

    # Random shuffle
    np.random.seed(1)
    train_sample = np.random.choice(np.arange(x_train.shape[0]), x_train.shape[0], replace=False)
    validate_sample = np.random.choice(np.arange(x_validate.shape[0]), x_validate.shape[0], replace=False)
    
    x_train = x_train[train_sample, ..., :3]
    y_train = y_train[train_sample]
    
    x_validate = x_validate[validate_sample, ..., :3]
    y_validate = y_validate[validate_sample]

    # Check for nans/infs
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(y_train))
    assert not np.any(np.isnan(x_validate))
    assert not np.any(np.isnan(y_validate))

    print("\n##############################")
    print(f"GENERATING MULTI-SCALE MODEL")
    print("##############################\n")

    ms_model = MultiScaleModel(y_train.shape[-1])
    ms_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                     loss=losses.categorical_crossentropy,
                     metrics=['accuracy'],
                     )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(os.path.join(save_directory, name + ".h5"),
                         monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    
    ms_model.fit(x_train, y_train, epochs=100,
                 validation_data=(x_validate, y_validate),
                 callbacks=[es, mc], batch_size=64)

    return ms_model

def train_dual_multiscale_model(train_data_tuple, validation_data_tuple, save_directory, name):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    x_train, y_train = train_data_tuple
    x_validate, y_validate = validation_data_tuple

    # Random shuffle
    np.random.seed(1)
    train_sample = np.random.choice(np.arange(x_train.shape[0]), x_train.shape[0], replace=False)
    validate_sample = np.random.choice(np.arange(x_validate.shape[0]), x_validate.shape[0], replace=False)
    
    x_train = x_train[train_sample, ..., :3]
    y_train = y_train[train_sample]
    
    x_validate = x_validate[validate_sample, ..., :3]
    y_validate = y_validate[validate_sample]

    # Check for nans/infs
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(y_train))
    assert not np.any(np.isnan(x_validate))
    assert not np.any(np.isnan(y_validate))

    print("\n##############################")
    print(f"GENERATING BENCHMARK OCNN MODEL")
    print("##############################\n")

    benchmark_model = BenchmarkModel(y_train.shape[-1])
    benchmark_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                            loss=losses.categorical_crossentropy,
                            metrics=['accuracy'],
                            )
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(os.path.join(save_directory, name + "_benchmark.h5"),
                         monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    benchmark_model.fit(x_train, y_train, epochs=100,
                        validation_data=(x_validate, y_validate),
                        callbacks=[es, mc], batch_size=64)

    print("\n##############################")
    print(f"GENERATING MULTI-SCALE MODEL")
    print("##############################\n")

    ms_model = MultiScaleModel(y_train.shape[-1])
    ms_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                     loss=losses.categorical_crossentropy,
                     metrics=['accuracy'],
                     )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(os.path.join(save_directory, name + ".h5"),
                         monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    
    ms_model.fit(x_train, y_train, epochs=100,
                 validation_data=(x_validate, y_validate),
                 callbacks=[es, mc], batch_size=64)

    return benchmark_model, ms_model

if __name__ == "__main__":
    # directory = "JDL_data/Kano"
    # xt = np.load(os.path.join(directory, 'X_train_192.npy')).astype(np.float32)[..., :-1]
    # xv = np.load(os.path.join(directory, 'X_validate_192.npy')).astype(np.float32)[..., :-1]
    # yt = to_categorical(np.load(os.path.join(directory, 'y_train_192.npy')))
    # yv = to_categorical(np.load(os.path.join(directory, 'y_validate_192.npy')))
    #
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         tf.config.experimental.set_visible_devices(gpus, 'GPU')
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         print(e)
    #
    # # Random shuffle
    # np.random.seed(1)
    # train_sample = np.random.choice(np.arange(xt.shape[0]), xt.shape[0], replace=False)
    # validate_sample = np.random.choice(np.arange(xv.shape[0]), xv.shape[0], replace=False)
    # xt = xt[train_sample]
    # yt = yt[train_sample]
    # xv = xv[validate_sample]
    # yv = yv[validate_sample]
    #
    # # Check for nans/infs
    # assert not np.any(np.isnan(xt))
    # assert not np.any(np.isnan(yt))
    # assert not np.any(np.isnan(xv))
    # assert not np.any(np.isnan(yv))
    #
    # model = MultiZoneModel(yt.shape[-1])
    # model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
    #               loss=losses.categorical_crossentropy,
    #               metrics=['accuracy'],
    #               )
    # save_directory = "models/multi_scale_ocnns"
    # name = "benchmark_192"
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)
    # mc = ModelCheckpoint(os.path.join(save_directory, name + ".h5"),
    #                      monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    #
    # model.fit(xt, yt, epochs=100,
    #           validation_data=(xv, yv),
    #           callbacks=[es, mc], batch_size=64)
    # # model.load_weights("models/multi_scale_ocnns/first_test.h5")
    #
    # for layer in model.layers:
    #     try:
    #         print(layer.name, layer.get_weights())
    #     except AttributeError:
    #         pass

    pass
