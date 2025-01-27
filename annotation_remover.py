
import os
from util import save_figure
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.filters import threshold_sauvola
import tensorflow as tf
import random
from train import resize_img, DIM
from ocr import OCR


class AnnotationClassifier:
    """
    Loads neural network to classify components 
    """

    def __init__(self, model_path, dim, pad, plot):
        """Constructor

        :param model_path: path to trained model
        :param dim: dimension each component should be resized to
        :param pad: whether to pad images
        :param plot: writes detected comment components to disk
        """
        self.model = tf.keras.models.load_model(model_path)
        self.dim = dim
        self.pad = pad
        self.plot = plot
        
        if self.plot:
            self.comp_path = "comps"
            self.plot_count = 0
            os.makedirs(self.comp_path, exist_ok=True)

    def predict(self, img):
        """Run prediction

        :param img: input image
        :return: True if component was a comment else False
        """
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
    """
    Extracts connected components (CC) from image
    """

    def __init__(self, img_path, min_area=100, max_area=5000, min_dim=10, 
                 max_dim=100, remove_ratio=0.15, plot=False):
        """Constructor (makes extraction call in constructor already)

        :param img_path: path to image
        :param min_area: minimum pixel area a CC can have, defaults to 100
        :param max_area: maximum pixel area a CC can have, defaults to 5000
        :param min_dim: minimum pixel dimension a CC can have, defaults to 10
        :param max_dim: maximum pixel dimension a CC can have, defaults to 100
        :param remove_ratio: controls the size of the discard area in center of image, defaults to 0.15
        :param plot: whether to make plots, defaults to False
        """
        self.img_path = img_path
        self.min_area = min_area
        self.max_area = max_area
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.remove_ratio = remove_ratio
        self.plot = plot

        self._extract()
        
    def _extract(self):
        """Extract using connectedComponentsWithStatsWithAlgorithm function
        """
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

        self.total_comp, self.pixel_labels, self.comp_info, _ = cv2.connectedComponentsWithStatsWithAlgorithm(self.img_bin, 4, cv2.CV_32S, cv2.CCL_GRANA)

    def components(self):
        """Creates a generator that provides info about each extracted component

        :yield: x-coordinate of CC, y-coordinate of CC, width of CC, height of CC, binary mask of CC, cropped CC
        """
        if self.plot:
            self.img_draw_color = np.zeros(shape=self.img_draw.shape, dtype=np.uint8)

        for i in range(1, self.total_comp): 
            
            area = self.comp_info[i, cv2.CC_STAT_AREA]
            x, y = self.comp_info[i, cv2.CC_STAT_LEFT], self.comp_info[i, cv2.CC_STAT_TOP]
            width, height = self.comp_info[i, cv2.CC_STAT_WIDTH], self.comp_info[i, cv2.CC_STAT_HEIGHT]
            
            if ((area > self.min_area) and (area < self.max_area) and 
                min(width, height) > self.min_dim and max(width, height) <= self.max_dim): 
                comp_mask = (self.pixel_labels == i).astype(np.uint8) * 255
                comp_cropped = comp_mask[y:y+height, x:x+width]

                if self.plot:
                    color = np.random.choice(range(256), size=3).astype(np.uint8)
                    self.img_draw_color[comp_mask == 255] = color

                yield x, y, width, height, comp_mask, comp_cropped


class AnnotationRemover:
    """
    Comment removing pipeline (initially called this annotation remover)
    """

    def __init__(self, component_extractor, model, num_chars=8, verbose=False, 
                 plot=True, use_ocr=False, ocr_thresh=0.9):
        """Constructor

        :param component_extractor: instance of ComponentExtractor
        :param model: instance of AnnotationClassifier
        :param num_chars: detection threshold (minimum number of bounding boxes that need pass through a line), defaults to 8
        :param verbose: print debug, defaults to False
        :param plot: whether to make debug plots, defaults to True
        :param use_ocr: whether to use TrOCR to transcribe comment, defaults to False
        :param ocr_thresh: experimental (tried with TrOCR), defaults to 0.9
        """
        self.component_extractor = component_extractor
        self.img_crop = self.component_extractor.img_org.copy()
        self.model = model
        self.num_chars = num_chars
        self.verbose = verbose
        self.plot = plot
        self.rows = None
        self.use_ocr = use_ocr
        
        if self.use_ocr:
            self.ocr = OCR(ocr_thresh)

        if self.plot:
            self.img_draw = copy.deepcopy(self.img_crop)

    def draw_zoom_in(self, ax, img, x1, y1, width, height, x_new, y_new,
                     zoom_factor=2.1, color="red", line_width=2):
        """This is a hard coded function for the test image to highlight the comment in red.
        You need to manually input the location and size of the comment

        :param ax: matplotlib axis to draw on
        :param img: raw image
        :param x1: x-coordinate of comment
        :param y1: y-coordinate of comment
        :param width: width of comment
        :param height: height of comment
        :param x_new: x-coordinate for larger comment image
        :param y_new: y-coordinate for larger comment image
        :param zoom_factor: how much to enlarge the comment, defaults to 2.1
        :param color: color of bounding box, defaults to "red"
        :param line_width: stroke width of bounding box, defaults to 2
        """
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

    def get_debug_drawing(self, show=True, debug=False, debug_save_name=""):
        """Makes a debug drawing to see individual decisions of components

        :param show: whether to call plt.show() from matplotlib, defaults to True
        :param debug: debug print, defaults to False
        :param debug_save_name: where to save debug drawing, defaults to ""
        """
        if not self.plot or self.rows is None:
            print("No plot available!")
            return
        
        # plot for component extractor
        fig, ax = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(8, 5))

        ax[0].set_title("a) Raw Input Image")
        self.draw_zoom_in(ax[0], self.component_extractor.img_org, 0, 1000, 75, 780, 200, 50)
        
        ax[1].imshow(self.component_extractor.img_draw)
        ax[1].set_title("b) Binarized\nImage")
        ax[2].imshow(self.component_extractor.img_draw_color)
        ax[2].set_title("c) Extracted Components\nand Discarded Area")
        ax[3].imshow(self.img_comps, cmap="gray")
        ax[3].set_title("d) Components\nPredicted as\nComments by CNN")

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
        num_cols = 5 if not debug else 4
        fig, ax = plt.subplots(1, num_cols, sharey=True, figsize=(10, 5))

        if not debug:
            ax[0].set_title("a) Raw Input Image")
            self.draw_zoom_in(ax[0], self.component_extractor.img_org, 0, 1000, 75, 780, 200, 50)
            n = 0
        else:
            n = -1

        ax[n+1].imshow(self.boxes)
        ax[n+1].set_title("a) Components\nDetected as\nComments by CNN")
        ax[n+2].imshow(self.rows_draw)
        ax[n+2].set_title(f"b) Max Horizontal\nPass Through")
        ax[n+3].imshow(self.cols_draw)
        ax[n+3].set_title(f"c) Max Vertical\nPass Through")
        ax[n+4].imshow(self.img_draw)
        ax[n+4].set_title("d) Crop Lines")

        fig.tight_layout()
        save_name = "slides_comp" if not debug else debug_save_name
        save_figure(save_name, show=show)
    
    def _find_crop_line(self):
        """Given the classifications of the components find the crop lines.
        Calculates maximum and vertical bounding box pass through.
        """
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

        if self.use_ocr:
            self.comment = self.img_crop[contour_y:contour_y+contour_height,
                                         contour_x:contour_x+contour_width]
            height_c, width_c, _ = self.comment.shape
            if height_c > width_c:
                self.comment = cv2.rotate(self.comment, cv2.ROTATE_90_CLOCKWISE)
            text, prob = self.ocr.run(self.comment)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(self.comment)
            quote = '"'
            ax.set_title(f"Detected text: {quote}{text}{quote}\nScore: {np.round(prob, 3)}")
            save_figure("ocr_example", fig=fig, show=False)
            fig, ax = plt.subplots()
            ax.imshow(self.comment)
            ax.set_xticks([])
            ax.set_yticks([])
            save_figure("comment_example", fig=fig, show=False)

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
        """Run comment removing pipeline

        :return: cropped image
        """
  
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="path to directory of images", default="../datasets/ICDAR2017_CLaMM_Training", type=str, required=False)
    parser.add_argument("--plot", help="plot first image only", type=str, default="yes", required=False)
    args = parser.parse_args()

    PLOT = args.plot == "yes"

    path = args.img_dir
    clean_img_path = "cleaned_imgs"
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
        annotation_remover = AnnotationRemover(component_extractor, model, plot=plot, verbose=True, use_ocr=plot)
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

