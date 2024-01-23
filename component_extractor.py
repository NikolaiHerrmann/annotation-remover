
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
import tensorflow as tf


class ComponentExtractor:

    def __init__(self, min_area=100, max_area=5000, min_dim=10, verbose=False):
        self.verbose = verbose
        self.min_area = min_area
        self.max_area = max_area
        self.min_dim = min_dim
        self.img_draw = []
        self.model = tf.keras.models.load_model("my_model.keras")

    def predict(self, img, dim=(30, 30)):
        img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        return self.model((img / 255).reshape(1, *dim, 1))[0]

    def extract(self, img_path):
        self.img_org = cv2.imread(img_path)
        self.img_gray = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2GRAY)
        
        #self.img_bin = cv2.threshold(self.img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        self.img_bin = 255 - (255 * (self.img_gray >= threshold_sauvola(self.img_gray)).astype(np.uint8))

        self.total_comp, self.pixel_labels, self.comp_info, _ = cv2.connectedComponentsWithStatsWithAlgorithm(self.img_bin, 4, cv2.CV_32S, cv2.CCL_GRANA) #cv2.connectedComponentsWithStats(self.img_bin, 4, cv2.CV_32S)
        print("Done")

        return self._get_comp_imgs()
    
    def get_drawing(self, show=True):
        if self.img_draw is None:
            print("No drawing made, turn on verbose.")
            return None
        
        if show:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.img_org)
            ax[1].imshow(self.img_draw, cmap="gray")
            plt.show()

        return self.img_draw

    def _get_comp_imgs(self):
        # make this faster and don't store
        comp_ls = []
  
        if self.verbose:
            self.img_draw = np.zeros(self.img_bin.shape, dtype=np.uint8) 
        
        for i in range(1, self.total_comp): 
            
            area = self.comp_info[i, cv2.CC_STAT_AREA]
            x, y = self.comp_info[i, cv2.CC_STAT_LEFT], self.comp_info[i, cv2.CC_STAT_TOP]
            width, height = self.comp_info[i, cv2.CC_STAT_WIDTH], self.comp_info[i, cv2.CC_STAT_HEIGHT] 
            
            if (area > self.min_area) and (area < self.max_area) and min(width, height) > self.min_dim: 
                comp_mask = (self.pixel_labels == i).astype(np.uint8) * 255
                comp_img = comp_mask[y:y+height, x:x+width]

                # plt.imshow(comp_img)
                # plt.show()
                # exit()
                if max(width, height) > 100:
                    continue
                #comp_ls.append(comp_img)
                prediction = (self.predict(comp_img).numpy() > 0.5)[0]
                if not prediction:
                    self.img_draw = cv2.bitwise_or(self.img_draw, comp_mask)

        return comp_ls



if __name__ == "__main__":
    path ="../datasets/annotations/IRHT_P_002497_ant.tif"
    path = "../datasets/ICDAR2017_CLaMM_Training/IRHT_P_000205.tif"
    #path = "../ICDAR2017_CLaMM_task2_task4/315556101_MS0894_0066.jpg"
    comp_extract = ComponentExtractor(verbose=True)
    comp_extract.extract(path)
    img = comp_extract.get_drawing()

