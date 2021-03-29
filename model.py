from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
INPUT_SHAPE = (2,)


def create_model(input_shape=INPUT_SHAPE):
    in_layer = Input(shape=input_shape)
    hid_layer = Dense(2, activation="relu")(in_layer)
    out_layer = Dense(2, activation="linear")(hid_layer)
    model = Model(in_layer, out_layer)
    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt, loss="mse")
    model.summary()
    return model


if __name__ == "__main__":
    create_model()
