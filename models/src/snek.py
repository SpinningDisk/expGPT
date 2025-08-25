from keras import layers, Model
import sys
sys.path.insert(0, "/mnt/data/dev/ML/lib/")
from SelectiveTransformation import SelectiveTransformation

class SnakeBrain():
    def CNN(self):
        inputs = layers.Input((2, 18, 18, 1, 3))
        inputs = layers.Reshape((2, 18, 18, 3))(inputs)
        conv1 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation="relu", padding="same")(inputs)
        conv2 = layers.Conv3D(64, kernel_size=(1, 3, 3), activation="relu", padding="same")(conv1)
        f = layers.Flatten()(conv2)
        d1 = layers.Dense(128, activation="relu")(f)
        d2 = layers.Dense(64, activation="relu")(d1)
        out = layers.Dense(4, activation="linear")(d2)
        return Model(inputs=inputs, outputs=out)
    def CNN_with_SelectiveTransformation(self):
        inputs = layers.Input((2, 18, 18, 3))
        transformed = SelectiveTransformation([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (2, 18, 18, 1, 3), (3, ), True)(inputs)
        inputs = layers.Reshape((2, 18, 18, 3))(transformed)
        conv1 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation="relu", padding="same")(inputs)
        conv2 = layers.Conv3D(64, kernel_size=(1, 3, 3), activation="relu", padding="same")(conv1)
        f = layers.Flatten()(conv2)
        d1 = layers.Dense(128, activation="relu")(f)
        d2 = layers.Dense(64, activation="relu")(d1)
        out = layers.Dense(4, activation="linear")(d2)
        return Model(inputs=inputs, outputs=out)



if __name__ == "__main__":
    import keras
    import numpy as np
    local_agent = SnakeBrain().CNN_with_SelectiveTransformation()
    data = np.expand_dims(np.arange(2*18*18*3).reshape((2, 18, 18, 3)), 0)
    out = local_agent(data)
    print(out)
    local_agent.save("test.keras")
    imported_agent = keras.models.load_model("test.keras")
    imported_agent(data)
    print(out)

