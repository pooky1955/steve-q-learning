from keras.layers import *
from keras.models import Model
def create_model():
    in_layer = Input(input_shape=(1))
    out_layer = Dense(1,activation="sigmoid")(in_layer)
    model = Model(in_layer,out_layer)
    model.compile(optimizer="adam",loss="binary_crossentropy")
    return model



