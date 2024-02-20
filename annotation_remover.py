
from util import plt_save
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
import tensorflow as tf
from train import resize_img, DIM


class AnnotationClassifier:

    def __init__(self, model_path, dim, pad):
        self.model = tf.keras.models.load_model(model_path)
        self.dim = dim
        self.pad = pad

    def predict(self, img):
        img = resize_img(img, size=self.dim, pad=self.pad)
        prediction = self.model(img.reshape(1, self.dim, self.dim, 1), training=False)[0]
        return (prediction.numpy() > 0.5)[0]


class ComponentExtractor:

    def __init__(self, img_path, min_area=100, max_area=5000, min_dim=10, max_dim=100, remove_ratio=0.15):
        self.img_path = img_path
        self.min_area = min_area
        self.max_area = max_area
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.remove_ratio = remove_ratio

        self._extract()
        
    def _extract(self):
        self.img_org = cv2.imread(self.img_path)
        self.img_gray = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2GRAY)
        self.shape = self.img_gray.shape
        self.height, self.width = self.shape
        
        self.img_bin = 255 - (255 * (self.img_gray >= threshold_sauvola(self.img_gray)).astype(np.uint8))

        if self.remove_ratio > 0:
            height_cutoff = int(self.height * self.remove_ratio)
            width_cutoff = int(self.width * self.remove_ratio)
            self.img_bin[height_cutoff:self.height-height_cutoff, width_cutoff:self.width-width_cutoff] = 0

        self.total_comp, self.pixel_labels, self.comp_info, _ = cv2.connectedComponentsWithStatsWithAlgorithm(self.img_bin, 4, cv2.CV_32S, cv2.CCL_GRANA)

    def components(self):
        for i in range(1, self.total_comp): 
            
            area = self.comp_info[i, cv2.CC_STAT_AREA]
            x, y = self.comp_info[i, cv2.CC_STAT_LEFT], self.comp_info[i, cv2.CC_STAT_TOP]
            width, height = self.comp_info[i, cv2.CC_STAT_WIDTH], self.comp_info[i, cv2.CC_STAT_HEIGHT]
            
            if ((area > self.min_area) and (area < self.max_area) and 
                min(width, height) > self.min_dim and max(width, height) <= self.max_dim): 
                comp_mask = (self.pixel_labels == i).astype(np.uint8) * 255
                comp_cropped = comp_mask[y:y+height, x:x+width]

                yield x, y, width, height, comp_mask, comp_cropped


class AnnotationRemover:

    def __init__(self, component_extractor, model, num_chars=8, verbose=False, plot=True):
        self.component_extractor = component_extractor
        self.img_crop = self.component_extractor.img_org.copy()
        self.model = model
        self.num_chars = num_chars
        self.verbose = verbose
        self.plot = plot
        self.rows = None
        if self.plot:
            self.img_draw = self.img_crop.copy()

    def get_debug_drawing(self, show=True, extra_show=True):
        if not self.plot or self.rows is None:
            print("No plot available!")
            return
        
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(self.component_extractor.img_org)
        ax[0, 1].imshow(self.img_draw)
        ax[0, 2].imshow(self.img_crop)
        ax[1, 0].imshow(self.img_comps, cmap="gray")
        ax[1, 1].imshow(self.rows, cmap="gray")
        ax[1, 2].imshow(self.cols, cmap="gray")

        if show:
            plt.show()

        if extra_show:
            fig, ax = plt.subplots(1, 3, figsize=(15, 8))
            ax[0].imshow(self.img_comps, cmap="gray")
            ax[0].set_title("Detected Annotations")
            ax[1].imshow(self.rows, cmap="gray")
            ax[1].set_title("Max horizontal pass through")
            ax[2].imshow(self.cols, cmap="gray")
            ax[2].set_title("Max vertical pass through")
            plt_save("debug")
    
    def _find_crop_line(self):
        height, width = self.cols.shape

        cols_sums = self.cols.sum(axis=0)
        rows_sums = self.rows.sum(axis=1)

        cols_best_line = np.argmax(cols_sums)
        cols_best_count = cols_sums[cols_best_line] / 255
        rows_best_line = np.argmax(rows_sums)
        rows_best_count = rows_sums[rows_best_line] / 255

        # times 2 since we have two sets of parallel lines for a rect bounding box
        if max(rows_best_count, cols_best_count) < (self.num_chars * 2):
            if self.verbose:
                print("No annotation found")
            return

        axis, is_row, dim = (self.rows, True, height) if rows_best_count > cols_best_count else (self.cols, False, width)

        # draw lines to find contour (not for debug)
        cv2.line(self.cols, (cols_best_line, 0), (cols_best_line, height - 1), (255, 255, 255), 1)
        cv2.line(self.rows, (0, rows_best_line), (width - 1, rows_best_line), (255, 255, 255), 1)

        # find largest contour
        contours, _ = cv2.findContours(axis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key = cv2.contourArea)
        contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(largest_contour)

        min_lines = contour_y, contour_y + contour_height if is_row else contour_x, contour_x + contour_width
        min_lines = np.array(min_lines)
        cut_line_idx = np.argmin(np.abs(min_lines - (dim / 2)))
        cut_line = min_lines[cut_line_idx]

        min_idx, max_idx = (cut_line, dim) if (dim / 2) > cut_line else (0, cut_line)

        self.img_crop = self.img_crop[min_idx:max_idx, :] if is_row else self.img_crop[:, min_idx:max_idx]

        if self.plot:
            cv2.rectangle(self.img_draw, (contour_x, contour_y),
                          (contour_x + contour_width, contour_y + contour_height), (0, 255, 0), 3)

    def remove(self):
  
        self.cols = np.zeros(self.component_extractor.shape, dtype=np.uint8)
        self.rows = np.zeros(self.component_extractor.shape, dtype=np.uint8)

        if self.plot:
            self.img_comps = np.zeros(self.component_extractor.shape, dtype=np.uint8)

        for x, y, width, height, comp_mask, comp_cropped in self.component_extractor.components():
            is_annotation = self.model.predict(comp_cropped) 
            if is_annotation:
                cv2.line(self.cols, (x, y), (x + width, y), (255, 255, 255), 1)
                cv2.line(self.cols, (x, y + height), (x + width, y + height), (255, 255, 255), 1)

                cv2.line(self.rows, (x, y), (x, y + height), (255, 255, 255), 1)
                cv2.line(self.rows, (x + width, y), (x + width, y + height), (255, 255, 255), 1)

                if self.plot:
                    self.img_comps = cv2.bitwise_or(self.img_comps, comp_mask)

        self._find_crop_line()

        return self.img_crop
    

if __name__ == "__main__":
    import glob
    import random
    import os
    import shutil
    from multiprocessing import Pool

    PLOT = False

    path = "../datasets/ICDAR2017_CLaMM_task1_task3"
    img_ls = glob.glob(os.path.join(path, "*.tif"))
    csv_ls = glob.glob(os.path.join(path, "*.csv"))
    
    if PLOT:
        random.shuffle(img_ls)
    else:
        clean_img_path = "../datasets/CLaMM_task1_task3_Clean"
        os.makedirs(clean_img_path, exist_ok=False)
        
        assert len(csv_ls) == 1
        csv_name = os.path.basename(csv_ls[0])
        shutil.copy(csv_ls[0], os.path.join(clean_img_path, csv_name))

    def run(img_path, plot=False):
        model = AnnotationClassifier("remover_model_v1_pad.keras", DIM, True)
        component_extractor = ComponentExtractor(img_path)
        annotation_remover = AnnotationRemover(component_extractor, model, plot=plot, verbose=True)
        cropped_img = annotation_remover.remove()

        if plot:
            annotation_remover.get_debug_drawing()
        else:
            file_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(clean_img_path, file_name), cropped_img)

    if PLOT:
        for path in img_ls:
            run(path, plot=True)
    else:
        with Pool() as pool:
            pool.map(run, img_ls)       

