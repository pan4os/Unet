
import os
import numpy as np
import cv2
from skimage import img_as_ubyte
#constants
from constants import IMAGE_DIR, MASK_DIR



def train_data(train_path, img_width, img_height):
    train_ids = next(os.walk(train_path))[1]
    x = np.zeros((len(train_ids), img_width, img_height, 3), dtype=np.uint8)
    y = np.zeros((len(train_ids), img_width, img_height, 1), dtype=np.bool)

    for i, id_ in enumerate(train_ids):
        path = train_path + id_
        img = cv2.imread(path + IMAGE_DIR + id_ + '.png')
        img = cv2.resize(img, (img_width, img_height))
        x[i] = img
        mask = np.zeros((img_width, img_height, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + MASK_DIR))[2]:
            mask_ = cv2.imread(path + MASK_DIR + mask_file, 0)
            mask_ = cv2.resize(mask_, (img_width, img_height))
            mask_ = mask_[:, :, np.newaxis]
            mask = np.maximum(mask, mask_)
        y[i] = mask
    return x, y


def test_data(test_path, img_width, img_height):
    test_ids = next(os.walk(test_path))[1]
    pathes = []
    x = np.zeros((len(test_ids), img_width, img_height, 3), dtype=np.uint8)
    sizes_test = []

    for i, id_ in enumerate(test_ids):
        path = test_path + id_
        img = cv2.imread(path + IMAGE_DIR + id_ + '.png')
        pathes.append(path + IMAGE_DIR + id_ + 'mask.png')
        sizes_test.append([img.shape[0], img.shape[1]])
        img = cv2.resize(img, (img_height, img_height))
        x[i] = img
    return x

def save_predicted_data(predicted_masks,folder):
    length = predicted_masks.shape[0]
    for i in range(length):
        rgb_pred = cv2.cvtColor(predicted_masks[i], cv2.COLOR_GRAY2RGB)
        os.chdir(folder)
        filename = str(i+1) + '.png'
        cv_image = img_as_ubyte(rgb_pred)
        cv2.imwrite(filename, cv_image)




