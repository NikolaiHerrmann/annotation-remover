
import os
import glob
from annotation_remover import AnnotationClassifier, AnnotationRemover, ComponentExtractor, DIM
import random
import shutil
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import SEED


TEST_PATH = "test_imgs_2"
CLAMM_PATH = "../datasets/ICDAR2017_CLaMM_Training"


def pick_n_random(n=6):
    """Picks n number of images from the CLaMM dataset (check that we haven't picked them before)

    :param n: number of images to pick, defaults to 6
    """
    annotation_path = "annotations/*.tif"
    clamm_path = os.path.join(CLAMM_PATH, "*.tif")

    train_imgs = set()
    for x in glob.glob(annotation_path):
        x = os.path.basename(x)
        x = x.rsplit("_", 1)[0]
        x += ".tif"
        train_imgs.add(x)

    clamm_imgs = glob.glob(clamm_path)
    test_imgs = set()

    prev_run = pd.read_csv("test_imgs_final.csv")["img"].to_list()

    os.makedirs(TEST_PATH, exist_ok=True)

    csv_img = []
    csv_state = []
    
    for i in tqdm(range(n)):
        while True:
            img = random.choice(clamm_imgs)
            img_base = os.path.basename(img)
            if (img_base in test_imgs) or (img_base in train_imgs) or (img_base in prev_run):
                continue
            test_imgs.add(img_base)
            break
        shutil.copy(img, os.path.join(TEST_PATH, img_base))
        csv_img.append(img_base)
        csv_state.append(1)

    df = pd.DataFrame({"img": csv_img, "correct": csv_state})
    df.to_csv("test_imgs_final_2.csv", index=False)


def run_test():
    """Run pipeline for an experiment
    """
    for x in tqdm(glob.glob(os.path.join(TEST_PATH, "*.tif"))):
        cropped_img = run_pipeline(x)
        new_name = x.rsplit(".", 1)[0] + "_cut.png"
        cv2.imwrite(new_name, cropped_img)


def run_pipeline(x, plot=False, debug_save_name=None, use_ocr=False):
    """Input an image into the pipeline

    :param x: input image
    :param plot: whether to plot image, defaults to False
    :param debug_save_name: where to save plot, defaults to None
    :param use_ocr: whether to apply the TrOCR model, defaults to False
    :return: cropped image
    """
    model = AnnotationClassifier("remover_model_v1_pad.keras", DIM, pad=True, plot=False)
    component_extractor = ComponentExtractor(x, plot=plot)
    annotation_remover = AnnotationRemover(component_extractor, model, plot=plot, verbose=True, use_ocr=use_ocr)
    cropped_img = annotation_remover.remove()

    if plot:
        annotation_remover.get_debug_drawing(debug=True, debug_save_name=debug_save_name, show=False)

    return cropped_img

def get_results(path="test_imgs_final.csv", plot=True):
    """Get results from experiments

    :param path: path where to read data from (CSV file), defaults to "test_imgs_final.csv"
    :param plot: whether to show bad examples, defaults to True
    """
    df = pd.read_csv(path)
    comment_imgs = df[df["comment"] == "yes"]
    non_comment_imgs = df[df["comment"] == "no"]

    print("\nStats:")
    print(np.unique(comment_imgs["correct"], return_counts=True))
    print(np.unique(non_comment_imgs["correct"], return_counts=True))
    
    if not plot:
        return

    # plot examples
    comment_mistakes = comment_imgs[comment_imgs["correct"] == "no"]

    for i, x in enumerate(comment_mistakes["img"]):
        print("Test example for", x)
        run_pipeline(os.path.join(CLAMM_PATH, x), plot=True, debug_save_name=f"fail_com{i}")
        break

    non_comment_mistakes = non_comment_imgs[non_comment_imgs["correct"] == "no"]

    for i, x in enumerate(non_comment_mistakes["img"]):
        print("Test example for", x)
        run_pipeline(os.path.join(CLAMM_PATH, x), plot=True, debug_save_name=f"fail_no-com{i}")
        if i == 2:
            break


if __name__ == "__main__":
    random.seed(SEED)
    #pick_n_random(50)
    #run_test()
    get_results(path="test_imgs_final_2.csv", plot=False)
