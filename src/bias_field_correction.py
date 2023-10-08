import SimpleITK as sitk
import sys
import os
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
from time import time

im = nib.load('/home/matthis/datasets/BraTS_2021/BraTS2021_00036/BraTS2021_00036_t1.nii.gz').get_fdata()
np.savez_compressed('/home/matthis/datasets/test.npz', im)
t=time()
s = np.load('/home/matthis/datasets/test.npz')
im = s['arr_0']
t1 = time()
print("numpy compressed time:", t1-t)

t=time()
im = np.load('/home/matthis/datasets/test.npy')
t1 = time()
print("numpy time:", t1-t)