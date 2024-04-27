
from util import save_figure
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.filters import threshold_sauvola
import tensorflow as tf
from train import resize_img, DIM


class AnnotationClassifier:

    def __init__(self, model_path, dim, pad, plot):
        self.model = tf.keras.models.load_model(model_path)
        self.dim = dim
        self.pad = pad
        self.plot = plot
        
        if self.plot:
            self.comp_path = "comps"
            self.plot_count = 0
            os.makedirs(self.comp_path, exist_ok=True)

    def predict(self, img):
        img = resize_img(img, size=self.dim, pad=self.pad)
        prediction = self.model(img.reshape(1, self.dim, self.dim, 1), training=False)[0]
        prediction = (prediction.numpy() > 0.5)[0]

        if prediction and self.plot:
            path = os.path.join(self.comp_path, f"c_{self.plot_count}.png")
            img = cv2.flip(img, 1) * 255
            cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.plot_count += 1

        return prediction


class ComponentExtractor:

    def __init__(self, img_path, min_area=100, max_area=5000, min_dim=10, 
                 max_dim=100, remove_ratio=0.15, plot=False):
        self.img_path = img_path
        self.min_area = min_area
        self.max_area = max_area
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.remove_ratio = remove_ratio
        self.plot = plot

        self._extract()
        
    def _extract(self):
        self.img_org = cv2.imread(self.img_path)
        self.img_gray = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2GRAY)
        self.shape = self.img_gray.shape
        self.height, self.width = self.shape
        
        self.img_bin = 255 - (255 * (self.img_gray >= threshold_sauvola(self.img_gray)).astype(np.uint8))

        if self.plot:
            self.img_draw = cv2.cvtColor(self.img_bin.copy(), cv2.COLOR_GRAY2RGB)

        if self.remove_ratio > 0:
            height_cutoff = int(self.height * self.remove_ratio)
            width_cutoff = int(self.width * self.remove_ratio)
            self.img_bin[height_cutoff:self.height-height_cutoff, width_cutoff:self.width-width_cutoff] = 0

            if self.plot:
                self.img_draw[height_cutoff:self.height-height_cutoff, width_cutoff:self.width-width_cutoff] = (0, 0, 0)

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

    def draw_zoom_in(self, ax, img, x1, y1, width, height, x_new, y_new,
                     zoom_factor=2.1, color="red", line_width=2):
        
        extract = img[y1:y1+height, x1:x1+width].copy()
        height_small, width_small, _ = extract.shape

        extract = cv2.resize(extract, (0, 0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        height_large, width_large, _ = extract.shape

        x_small, y_small = x1, y1        

        x_large, y_large = x_new, y_new
        img[y_large:y_large+height_large, x_large:x_large+width_large, :] = extract
        
        rect_large = Rectangle((x_large, y_large), width_large, height_large, linewidth=line_width, edgecolor=color, facecolor="none")
        rect_small = Rectangle((x_small, y_small), width_small, height_small, linewidth=line_width, edgecolor=color, facecolor="none")

        ax.plot((x_small, x_large), (y_small, y_large), color=color, linewidth=line_width)
        ax.plot((x_small + width_small, x_large + width_large), (y_small + height_small, y_large + height_large), color=color, linewidth=line_width) 
        ax.imshow(img)
        ax.add_patch(rect_large)
        ax.add_patch(rect_small)

    def get_debug_drawing(self, show=True):
        if not self.plot or self.rows is None:
            print("No plot available!")
            return
        
        # plot for component extractor
        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)

        ax[0].set_title("a) Raw Input Image")
        self.draw_zoom_in(ax[0], self.component_extractor.img_org, 0, 1000, 75, 780, 200, 50)
        
        ax[1].imshow(self.component_extractor.img_draw)
        ax[1].set_title("b) Binarized Image\nwith Filtered Components\nand Discarded Area")
        ax[2].imshow(self.img_comps, cmap="gray")
        ax[2].set_title("c) Components\nPredicted as\nComments by CNN")

        fig.tight_layout()
        save_figure("rm_components", show=False)
        
        fig, ax = plt.subplots(1, 3, sharey=True)
        ax[0].imshow(self.boxes)
        ax[0].set_title("a) Bounding boxes\nof Components")
        ax[1].imshow(self.rows_draw)
        ax[1].set_title(f"b) Max Horizontal\nPass Through")
        ax[2].imshow(self.cols_draw)
        ax[2].set_title(f"c) Max Vertical\nPass Through")

        fig.tight_layout()
        save_figure("rm_lines", show=False)

        fig, ax = plt.subplots(1, 3, sharey=True, subplot_kw=dict(aspect="equal"),
                               gridspec_kw=dict(width_ratios=[2, 2, 1.9]))
        ax[0].imshow(self.contour_draw)
        ax[0].set_title("a) Crop Lines")
        ax[1].imshow(self.img_draw)
        ax[1].set_title("b) Crop Lines Overlaid")
        ax[2].imshow(self.img_crop)
        ax[2].set_title("c) Final Cropped\nOutput Image")

        fig.tight_layout()
        save_figure("rm_crop_lines", show=False)

        # For slides
        fig, ax = plt.subplots(1, 5, sharey=True, figsize=(10, 5))

        ax[0].set_title("a) Raw Input Image")
        self.draw_zoom_in(ax[0], self.component_extractor.img_org, 0, 1000, 75, 780, 200, 50)
        ax[1].imshow(self.boxes)
        ax[1].set_title("a) Components\nDetected as\nComments by CNN")
        ax[2].imshow(self.rows_draw)
        ax[2].set_title(f"b) Max Horizontal\nPass Through")
        ax[3].imshow(self.cols_draw)
        ax[3].set_title(f"c) Max Vertical\nPass Through")
        ax[4].imshow(self.img_draw)
        ax[4].set_title("d) Crop Lines")

        fig.tight_layout()
        save_figure("slides_comp", show=show)
    
    def _find_crop_line(self):
        height, width = self.cols.shape

        cols_sums = self.cols.sum(axis=0)
        rows_sums = self.rows.sum(axis=1)

        cols_best_line = np.argmax(cols_sums)
        self.cols_best_count = cols_sums[cols_best_line] / 255
        rows_best_line = np.argmax(rows_sums)
        self.rows_best_count = rows_sums[rows_best_line] / 255

        # times 2 since we have two sets of parallel lines for a rect bounding box
        if max(self.rows_best_count, self.cols_best_count) < (self.num_chars * 2):
            if self.verbose:
                print("No annotation found")
            return

        axis, is_row, dim = (self.rows, True, height) if self.rows_best_count > self.cols_best_count else (self.cols, False, width)

        if self.plot:
            axis_draw = axis.copy()

        # draw lines to find contour (not for debug)
        cv2.line(self.cols, (cols_best_line, 0), (cols_best_line, height - 1), (255, 255, 255), 1)
        cv2.line(self.rows, (0, rows_best_line), (width - 1, rows_best_line), (255, 255, 255), 1)

        if self.plot:
            color = (255, 0, 0)
            self.cols_draw = cv2.cvtColor(self.cols, cv2.COLOR_GRAY2BGR)
            self.rows_draw = cv2.cvtColor(self.rows, cv2.COLOR_GRAY2BGR)
            cv2.line(self.cols_draw, (cols_best_line, 0), (cols_best_line, height - 1), color, self.thickness)
            cv2.line(self.rows_draw, (0, rows_best_line), (width - 1, rows_best_line), color, self.thickness)

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
            def draw_rect(x):
                cv2.rectangle(x, (contour_x, contour_y),
                              (contour_x + contour_width, contour_y + contour_height), (0, 255, 0), 3)
                return x
            
            draw_rect(self.img_draw)

            self.contour_draw = np.zeros(axis.shape, dtype=np.uint8)
            y_s, y_e = contour_y, contour_y + contour_height
            x_s, x_e = contour_x, contour_x + contour_width
            self.contour_draw[y_s:y_e, x_s:x_e] = axis_draw[y_s:y_e, x_s:x_e]

            self.contour_draw = draw_rect(cv2.cvtColor(self.contour_draw, cv2.COLOR_GRAY2RGB))

    def remove(self):
  
        self.cols = np.zeros(self.component_extractor.shape, dtype=np.uint8)
        self.rows = np.zeros(self.component_extractor.shape, dtype=np.uint8)
        self.boxes = self.component_extractor.img_org.copy()

        if self.plot:
            self.img_comps = np.zeros(self.component_extractor.shape, dtype=np.uint8)

        for x, y, width, height, comp_mask, comp_cropped in self.component_extractor.components():
            is_annotation = self.model.predict(comp_cropped) 
            if is_annotation:
                self.thickness = 5 if self.plot else 1
                color = (255, 255, 255)

                cv2.line(self.cols, (x, y), (x + width, y), color, self.thickness)
                cv2.line(self.cols, (x, y + height), (x + width, y + height), color, self.thickness)

                cv2.line(self.rows, (x, y), (x, y + height), color, self.thickness)
                cv2.line(self.rows, (x + width, y), (x + width, y + height), color, self.thickness)

                if self.plot:
                    cv2.rectangle(self.boxes, (x, y), (x + width, y + height), (0, 255, 0), self.thickness) 
                    self.img_comps = cv2.bitwise_or(self.img_comps, comp_mask)

        self._find_crop_line()

        return self.img_crop
    

if __name__ == "__main__":
    import glob
    import random
    import os
    import shutil
    from multiprocessing import Pool

    PLOT = True

    path = "../datasets/ICDAR2017_CLaMM_Training"
    clean_img_path = ""
    img_exts = ["tif", "jpg", "JPG"]

    img_ls = []
    for ext in img_exts:
        img_ls += glob.glob(os.path.join(path, "*." + ext))

    csv_ls = glob.glob(os.path.join(path, "*.csv"))
    
    if PLOT:
        random.shuffle(img_ls)
        random.shuffle(img_ls)
    else:
        os.makedirs(clean_img_path, exist_ok=False)
        
        assert len(csv_ls) == 1
        dir_name = os.path.basename(clean_img_path)
        shutil.copy(csv_ls[0], os.path.join(clean_img_path, "@" + dir_name + ".csv"))

    def run(img_path, plot=False):
        model = AnnotationClassifier("remover_model_v1_pad.keras", DIM, pad=True, plot=True)
        component_extractor = ComponentExtractor(img_path, plot=plot)
        annotation_remover = AnnotationRemover(component_extractor, model, plot=plot, verbose=True)
        cropped_img = annotation_remover.remove()

        if plot:
            annotation_remover.get_debug_drawing()
        else:
            file_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(clean_img_path, file_name), cropped_img)

    if PLOT:
        for path in img_ls:
            print("Running pipe on:", path)
            run(path, plot=True)
            exit()
    else:
        with Pool() as pool:
            pool.map(run, img_ls)       

