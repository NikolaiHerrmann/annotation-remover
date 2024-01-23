
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import glob
import cv2
import numpy as np
import random

random.seed(42)
np.random.seed(42)




def pre_process(img_path, dim=(30, 30)):
    img = tf.keras.utils.load_img(img_path)#, cv2.IMREAD_GRAYSCALE)
    img = tf.image.resize_with_pad(img, 60, 60, antialias=True)
    img = tf.image.rgb_to_grayscale(img)
    # print(np.unique(img))
    # exit()
    #img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    img = img / 255
    return img


ant_ls = glob.glob("imgs/ant_aug/*")
text_ls = glob.glob("imgs/ant_text/*")
random.shuffle(text_ls)
text_ls = text_ls[:len(ant_ls)]


print(len(text_ls))
print(len(ant_ls)) #2020
X = [pre_process(img) for img in ant_ls] + [pre_process(img) for img in text_ls]
X = np.array(X)
y = ([0] * len(ant_ls)) + ([1] * len(ant_ls))
y = np.array(y).reshape(-1, 1)
print(X.shape, y.shape)

X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print(X_test.shape, X_train.shape)

model = tf.keras.Sequential([
    layers.Input(shape=(60, 60, 1)),
    layers.Conv2D(64, (5, 5), activation='relu', padding='same', strides=1),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(32, (5, 5), activation='relu', padding='same', strides=1),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')

    # layers.Input(shape=(30, 30, 1)),
    # layers.Conv2D(64, (5, 5), activation='relu', padding='same', strides=1),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=1),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Dropout(0.3),
    # layers.Flatten(),
    # layers.Dense(128, activation='relu'),
    # layers.Dense(128, activation='relu'),
    # layers.Dropout(0.5),
    # layers.Dense(1, activation='sigmoid')
    
    # layers.Input(shape=(30, 30, 1)),
    # layers.Conv2D(64, (5, 5), activation='relu', padding='same', strides=1),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Conv2D(128, (5, 5), activation='relu', padding='same', strides=1),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=1),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=1),
    # layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=1),
    # layers.MaxPooling2D((2, 2), padding='same'),

    # layers.Flatten(),
    # layers.Dense(512, activation='relu'),
    # layers.Dropout(0.5),
    # layers.Dense(512, activation='relu'),
    # layers.Dropout(0.5),
    # layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, shuffle=True, validation_data=(X_test, y_test), batch_size=64)
model.save("my_model_2.keras")