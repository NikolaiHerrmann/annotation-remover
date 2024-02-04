
from util import SEED
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import json
import os


DIM = 31
PAD = True


def resize_img(img, size=DIM, pad=PAD):
    if not pad:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        height, width = img.shape
        max_dim = np.max(img.shape)
        factor = size / max_dim

        img = cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        height, width = img.shape

        img_padded = np.zeros((size, size), dtype=np.uint8)
        offset_y = (size - height) // 2
        offset_x = (size - width) // 2
        img_padded[offset_y:offset_y+height, offset_x:offset_x+width] = img
        img = img_padded

    img = img / 255

    return img


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return resize_img(img)


def get_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(31, 31, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=1),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=1),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=1),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def get_summary(model):
    with open("model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def load_data(path, is_train):
    set_type = "train" if is_train else "test"
    ant_ls = glob.glob(os.path.join(path, set_type, "ant/*"))
    text_ls = glob.glob(os.path.join(path, set_type, "text/*"))

    size = len(ant_ls)
    assert size == len(text_ls)

    X = [read_img(img) for img in ant_ls] + [read_img(img) for img in text_ls]
    X = np.array(X)
    y = ([1] * size) + ([0] * size)
    y = np.array(y).reshape(-1, 1)

    X, y = shuffle(X, y, random_state=SEED)
    return X, y


def run_model(X_train, X_val, y_train, y_val, plot_loss=False):
    model = get_model()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=["accuracy"])

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=20, shuffle=True, validation_data=(X_val, y_val), batch_size=64, callbacks=[callback])
    
    y_val_pred = model.predict(X_val)
    y_val_pred = (y_val_pred > 0.5).astype(np.int64)

    metrics = [accuracy_score, f1_score, recall_score, precision_score]
    scores = []

    for m in metrics:
        scores.append(m(y_val, y_val_pred))
    print(scores)
    
    if plot_loss:
        model.save("remover_model_v1_pad.keras")
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        json.dump(history.history, open("history.json", "w"), indent=4)

        plt.plot(loss, label="Training")
        plt.plot(val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross Entropy")
        plt.title("Annotation Remover Loss Curve")
        plt.legend()

        name = "loss_curve_2"
        plt.savefig(name + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(name + ".pdf", bbox_inches="tight")

    return scores


def run_kfold(X, y):
    skf = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
    scores = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index, :], X[test_index, :]
        y_train, y_val = y[train_index, :], y[test_index, :]

        score = run_model(X_train, X_val, y_train, y_val)
        scores.append(score)

    print(scores)
    print("---")
    print("accuracy, f1, recall, precision")
    print("Means", np.mean(np.array(scores), axis=0))
    print("Std", np.std(np.array(scores), axis=0))


if __name__ == "__main__":
    X, y = load_data("imgs", is_train=True)

    run_kfold(X, y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    run_model(X_train, X_val, y_train, y_val, plot_loss=True)
