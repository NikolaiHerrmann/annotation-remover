
import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

SEED = 42
FIGURE_PATH = "figs"

def set_seed():
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

set_seed()


def save_figure(title, fig=None, fig_dir=FIGURE_PATH, show=False, pdf=True, png=True, dpi=400):
    if not fig:
        fig = plt.gcf()
    if png:
        fig.savefig(os.path.join(fig_dir, title + ".png"), dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(fig_dir, title + ".pdf"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()