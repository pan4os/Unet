
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from metrics import dice_coef, dice_loss

def build_model(image_height, image_width, image_channels):
    start_neurons = 16

    input_layer = Input((image_height, image_width, image_channels))
    input_normalized = Lambda(lambda x: x / 255)(input_layer)

    conv1 = Conv2D(start_neurons, (3, 3), activation='relu', padding="same")(input_normalized)
    conv1 = Conv2D(start_neurons, (3, 3), activation='relu', padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.2)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.2)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.2)(pool4)

    conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(conv5)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding='same')(conv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.2)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.2)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.2)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.2)(uconv1)
    uconv1 = Conv2D(start_neurons, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_loss, dice_coef])
    return model