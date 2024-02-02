
import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def plt_save(title):
    plt.savefig(title + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(title + ".pdf", bbox_inches="tight")