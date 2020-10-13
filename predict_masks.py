import random

from data_loader import *
from constants import*
from model import build_model
import matplotlib.pyplot as plt
from skimage.io import  imshow

X_test = test_data(TEST_PATH,IMG_WIDTH,IMG_HEIGHT)
model  = build_model(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
model.load_weights('model-unet-dsbowl2018-1.h5')
y_pred = model.predict(X_test)
if not (os.path.exists(CUR_DIR + "\\" + FOLDER_NAME)):
    os.mkdir(FOLDER_NAME)
PRED_DIR = CUR_DIR +'\\' + FOLDER_NAME
for i in range(5):

    ix = random.randint(0, len(X_test))
    imshow(X_test[ix])
    plt.show()
    imshow(np.squeeze(y_pred[ix]))
    plt.show()
save_predicted_data(y_pred,PRED_DIR)

