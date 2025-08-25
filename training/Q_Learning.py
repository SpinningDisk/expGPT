import keras
import numpy as np




class onPolicy_QLearning():
    def __init__(self, model_output_shape, model, gamma: int=0.95, epsilon: int=1.0):
        gamma = 0.95
        epsilon = 1.0
        model
class offPolicy_QLearning(onPolicy_QLearning):
    def __init__(self, model_output_shape, model, pool_size, gamma: int=0.95, epsilon: int=1.0,):
        super().__init__(model_output_shape, gamma, epsilon, model)
        print(f"{'\033[1;32m'}intialized {'\033[1;36m'}off policy q learning {'\033[1;32m'}with:\n\tmodel output shape: {'\033[1;30m'}{model_output_shape}{'\033[1;32m'}\n\tgamma: {'\033[1;30m'}{gamma}{'\033[1;32m'}\n\tepsilon: {'\033[1;30m'}{epsilon}{'\033[1;32m'}\n\tpool length: {'\033[1;30m'}{pool_size}{'\033[0m'}")

if __name__ == "__main__":
    def return_model():
        input = keras.layers.Input((2, 18, 18, 3))
        conv3_1 = keras.layers.Conv3D(filters=32, kernel_size=(2, 3, 3), padding="same", activation="relu")(input)
        conv3_2 = keras.layers.Conv3D(filters=64, kernel_size=(2, 3, 3), padding="valid",activation="relu")(conv3_1)
        conv3_2 = keras.layers.Reshape((16, 16, 64))(conv3_2)
        conv2_1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(conv3_2)
        print(conv2_1.shape)
        return keras.Model(inputs=input, outputs=conv3_1)
    model = return_model()
    # rithm = offPolicy_QLearning((4, ))
