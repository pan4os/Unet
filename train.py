
from model import build_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from data_loader import *
from constants import *

def train(model,
          X_train, y_train,
          validation_split: float = 0.1,
          batch_size: int = 16,
          epochs: int = 50) -> History:
    earlystopper = EarlyStopping(monitor='val_dice_coef', mode='max', patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-unet-dsbowl2018-1.h5', verbose=1,
                                   monitor='val_dice_coef', mode='max', save_best_only=True)
    return model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                     callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':

    X_train,y_train = train_data(TRAIN_PATH,IMG_WIDTH,IMG_HEIGHT)

    X_test = test_data(TEST_PATH, IMG_WIDTH, IMG_HEIGHT)

    model = build_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    #fitted_model = train(model,X_train,y_train)
    earlystopper = EarlyStopping(monitor='val_dice_coef', mode='max', patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-unet-dsbowl2018-1.h5', verbose=1,
                                   monitor='val_dice_coef', mode='max', save_best_only=True)
    results = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=20,
                        callbacks=[earlystopper, checkpointer])