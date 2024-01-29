
from util import SEED
import os
import numpy as np
import glob
import cv2
import random
import matplotlib.pyplot as plt
from annotation_remover import ComponentExtractor
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def img_basename(path):
    return os.path.basename(path).rsplit(".", 1)[0]


def save_img(path, name, arr):
    plt.imsave(os.path.join(path, name), arr, cmap="gray")


def extract(save_path, read_path, search_card):
    os.makedirs(save_path, exist_ok=True)

    img_paths = glob.glob(os.path.join(read_path, search_card))

    for path in tqdm(img_paths):
        comp_extract = ComponentExtractor(path, max_dim=5000, remove_ratio=0)

        img_name = img_basename(path)
        for i, (_, _, _, _, _, comp) in enumerate(comp_extract.components()):
            comp_name = img_name + f"_{i}.ppm"
            save_img(save_path, comp_name, comp)


def save_images(img_paths, save_path, augment, examine_shapes=False):
    os.makedirs(save_path, exist_ok=True)
    count = 0
    shapes = []

    for path in img_paths:
        img_name = img_basename(path)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        shapes.append(img.shape)
        save_img(save_path, img_name + "_org.ppm", img)
        count += 1
        if augment:
            for i in range(3):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                save_img(save_path, img_name + f"_aug{i}.ppm", img)
                count += 1

    if examine_shapes:
        shapes = np.array(shapes)
        print("--")
        print("Mean shape:", np.mean(shapes, axis=0))
        print("Median shape:", np.median(shapes, axis=0))

    return count


def make_splits(ant_path, text_path, test_path, train_path):
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)

    ant_ls = glob.glob(ant_path)
    text_ls = glob.glob(text_path)

    print("Num annotation examples:", len(ant_ls))
    print("Num text examples:", len(text_ls))

    ant_train, ant_test = train_test_split(ant_ls, test_size=0.1, random_state=SEED)

    ant_train_count = save_images(ant_train, os.path.join(train_path, "ant"), augment=True, examine_shapes=True)
    ant_test_count = save_images(ant_test, os.path.join(test_path, "ant"), augment=False)

    random.shuffle(text_ls)
    text_train = text_ls[:ant_train_count] # randomly take same amount of annotations to balance dataset
    text_test = text_ls[ant_train_count:ant_train_count+ant_test_count]

    text_train_count = save_images(text_train, os.path.join(train_path, "text"), augment=False, examine_shapes=True)
    text_test_count = save_images(text_test, os.path.join(test_path, "text"), augment=False)

    assert ant_train_count == text_train_count and ant_test_count == text_test_count, "Splitting went wrong!"
    print("\nTrain_count:", ant_train_count, "Test_count:", ant_test_count)
            

if __name__ == "__main__":

    # Run extraction
    # extract("imgs/ant_raw", "annotations", "*_ant.*")
    # extract("imgs/text_raw", "annotations", "*_text.*")

    # Run to make dataset splits
    make_splits("imgs/ant_raw/*", "imgs/text_raw/*", "imgs/test", "imgs/train")