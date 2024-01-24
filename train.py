
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import json
import random
import os


SEED = 42


def set_seed():
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def pre_process(img_path, dim=(30, 30)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    img = img / 255
    return img


if __name__ == "__main__":
    set_seed()
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

    X, y = shuffle(X, y, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    print(X_test.shape, X_train.shape)

    model = tf.keras.Sequential([
        layers.Input(shape=(30, 30, 1)),
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
    ])

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=15, shuffle=True, validation_data=(X_test, y_test), 
                        batch_size=64, callbacks=[callback])
    #model.save("my_model_2.keras")

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    json.dump(history.history, open("history.json", "w"), indent=4)

    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Annotation Remover Loss Curve")
    plt.legend
    plt.legend()

    name = "loss_curve"
    plt.savefig(name + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(name + ".pdf", bbox_inches="tight")