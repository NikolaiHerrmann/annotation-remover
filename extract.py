
import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from component_extractor import ComponentExtractor
from tqdm import tqdm
import random

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


def img_basename(path):
    return os.path.basename(path).rsplit(".", 1)[0]


def save_img(path, name, arr):
    plt.imsave(os.path.join(path, name), arr, cmap="gray")


def extract(save_path, read_path, search_card):
    os.makedirs(save_path, exist_ok=True)

    img_paths = glob.glob(os.path.join(read_path, search_card))
    comp_extract = ComponentExtractor(verbose=False)

    for path in tqdm(img_paths):
        comps = comp_extract.extract(path)

        img_name = img_basename(path)
        for i, comp in enumerate(comps):
            comp_name = img_name + f"_{i}.ppm"
            save_img(save_path, comp_name, comp)


def augment(save_path, read_path):
    os.makedirs(save_path, exist_ok=True)

    img_paths = glob.glob(os.path.join(read_path, "*.ppm"))
    for path in img_paths:
        img_name = img_basename(path)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        save_img(save_path, img_name + "_org.ppm", img)
        for i in range(3):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            save_img(save_path, img_name + f"_aug{i}.ppm", img)
            

if __name__ == "__main__":
    #extract("imgs/ant_raw", "../datasets/annotations", "*_ant.*")
    extract("imgs/ant_text", "../datasets/annotations", "*_text.*")
    #augment("imgs/ant_aug", "imgs/ant_raw")