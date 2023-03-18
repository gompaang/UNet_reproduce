import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 0. data load
dir_data = './data'
name_input = 'train-volume.tif'
name_label = 'train-labels.tif'

img_input = Image.open(os.path.join(dir_data, name_input))
img_label = Image.open(os.path.join(dir_data, name_label))
ny, nx = img_label.size
nframe = img_label.n_frames #30


# 1. data split
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train_input = os.path.join(dir_data, 'train/input')
dir_save_train_label = os.path.join(dir_data, 'train/label')
dir_save_val_input = os.path.join(dir_data, 'val/input')
dir_save_val_label = os.path.join(dir_data, 'val/label')
dir_save_test_input = os.path.join(dir_data, 'test/input')
dir_save_test_label = os.path.join(dir_data, 'test/label')


# 2. frame random shuffle
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)


# 3. train data's (input, label)
offset_nframe = 0
for i in range(nframe_train):
    img_input.seek(id_frame[i+offset_nframe])
    img_label.seek(id_frame[i+offset_nframe])
    input_ = np.asarray(img_input)
    label_ = np.asarray(img_label)
    np.save(os.path.join(dir_save_train_input, 'input_%03d.npy' % i), input_)
    np.save(os.path.join(dir_save_train_label, 'label_ %03d.npy' % i), label_)


# 4. validation data's (input, label)
offset_nframe = nframe_train
for i in range(nframe_val):
    img_input.seek(id_frame[i+offset_nframe])
    img_label.seek(id_frame[i+offset_nframe])
    input_ = np.asarray(img_input)
    label_ = np.asarray(img_label)
    np.save(os.path.join(dir_save_val_input, 'input_%03d.npy' % i), input_)
    np.save(os.path.join(dir_save_val_label, 'label_ %03d.npy' % i), label_)


# 5. test data's (input, lable)
offset_nframe = nframe_train + nframe_val
for i in range(nframe_val):
    img_input.seek(id_frame[i+offset_nframe])
    img_label.seek(id_frame[i+offset_nframe])
    input_ = np.asarray(img_input)
    label_ = np.asarray(img_label)
    np.save(os.path.join(dir_save_test_input, 'input_%03d.npy' % i), input_)
    np.save(os.path.join(dir_save_test_label, 'label_ %03d.npy' % i), label_)