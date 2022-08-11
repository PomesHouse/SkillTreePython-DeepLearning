# data_loader.py
import tensorflow as tf
import numpy as np

# 정규화
# minmax = 0~1
# standard norma: -1 ~ 1

def mnist_loader(standard = False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, 3)


    x_train = x_train /255.  # 0 ~ 1
    x_test = x_test / 255.   # 0 ~ 1

    if standard:
        x_train = (x_train *2) -1
        x_test = (x_test *2) -1

    return x_train, y_train, x_test, y_test

def fmnist_loader(stadard = False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, 3)

    x_train = x_train /255. 
    x_test = x_test / 255.
    if standard:
        x_train = (x_train *2) -1
        x_test = (x_test *2) -1

    return x_train, y_train, x_test, y_test

def cifar10_loader(standard = False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train /255. 
    x_test = x_test / 255.
    if standard:
        x_train = (x_train *2) -1
        x_test = (x_test *2) -1
    return x_train, y_train, x_test, y_test

import matplotlib.pyplot as plt
import numpy as np

def display_sample_img(samples, standard = False, cmap='gray_r'):
  plt.figure(figsize=(15,3))
  for i,sample in enumerate(samples):
    if standard:
        sample = ( sample  + 1. ) / 2. # 다시 -1 ~ 1을 0 ~ 1
        sample = np.clip(sample, 0, 1) # 0 ~ 1 을 벗어나면 자르기
    if i==10:break
    plt.subplot(1,10,i+1)
    if sample.shape[-1]==1: plt.imshow(sample[:,:,0], cmap=cmap)
    else: plt.imshow(sample)
    plt.xticks([]);plt.yticks([])
  plt.show()