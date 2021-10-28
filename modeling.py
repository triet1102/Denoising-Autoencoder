from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model

image_size = 64


def create_model():
    input_img = Input(shape=(image_size, image_size, 1), name="image_input")

    # encoder
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv1")(input_img)
    x = MaxPooling2D((2, 2), padding="same", name="pool1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv2")(x)
    x = MaxPooling2D((2, 2), padding="same", name="pool2")(x)

    # decoder
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv3")(x)
    x = UpSampling2D((2, 2), name="upsample1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv4")(x)
    x = UpSampling2D((2, 2), name="upsample2")(x)
    x = Conv2D(1, (3, 3), activation="relu", padding="same", name="Conv5")(x)

    # model
    model = Model(inputs=input_img, outputs=x)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    return model
