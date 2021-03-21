from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model

def create_model():
    in_layer = Input(shape=(2))
    out_layer = Dense(1,activation="sigmoid")(in_layer)
    model = Model(in_layer,out_layer)
    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt,loss="binary_crossentropy")
    model.summary()
    return model

if __name__ == "__main__":
    create_model()

