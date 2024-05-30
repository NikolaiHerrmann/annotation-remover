# Comment Remover for ICDAR 2017 CLaMM Dataset

- A classifier was trained to crop out handwritten/printed comments in the CLaMM 2016 [1] and 2017 [2] datasets. 

- CNN Model is adapted from [3].

- Trained on 28 images from the 2016 dataset.

![](example.png)

# Run

Install all Python requirements:
```
pip install -r requirements.txt
```

To run comment remover run:
```
python annotation_remover.py --img_dir [clamm_img_dir] --plot [yes/no]
```
`--img_dir` specifies were the clamm images are and `--plot` if we should run an example on the first image found while making various debug plots.

To train the model see the `train.py` file and its main function for examples. However, model weights are provided. Model `remover_model_v1_pad.keras` was trained with padding while `remover_model_v1.keras` was not.

# Overview

- [annotation_remover.py](annotation_remover.py): includes all classes to run the pipeline

- [extract.py](extract.py): extract contours from 28 cropped training images

- [train.py](train.py): train and test the CNN

- [test.py](test.py): tests the pipeline on 50 random CLaMM images

- [ocr.py](ocr.py): uses TrOCR to transcribe a comment

- [test_imgs_final.csv](test_imgs_final.csv) and [test_imgs_final_2.csv](test_imgs_final_2.csv): experiment results from 50 random images

- [history.json](history.json), [scores.npy](scores.npy) and [model_summary.txt](model_summary.txt): epoch results and benchmarking from CNN training

# References

[1] Cloppet, F., Eglin, V., Stutzmann, D., & Vincent, N. (2016, October). ICFHR2016 competition on the classification of medieval handwritings in latin script. In 2016 15th International Conference on Frontiers in Handwriting Recognition (ICFHR) (pp. 590-595). IEEE.

[2] Cloppet, F., Eglin, V., Helias-Baron, M., Kieu, C., Vincent, N., & Stutzmann, D. (2017, November). Icdar2017 competition on the classification of medieval handwritings in latin script. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) (Vol. 1, pp. 1371-1376). IEEE.

[3] Ahamed, P., Kundu, S., Khan, T., Bhateja, V., Sarkar, R., & Mollah, A. F. (2020). Handwritten Arabic numerals recognition using convolutional neural network. Journal of Ambient Intelligence and Humanized Computing, 11, 5445-5457.