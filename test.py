import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2


img = cv2.imread("C:/Users/CWJ/Desktop/dataset/videos/Pairs_Free_seg1/output_0001.jpg")
w,h,c = img.shape
if w < 226 or h < 226:
    d = 226.-min(w,h)
    sc = 1+d/min(w,h)
    img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
else:
    # 如果图像已经大于等于目标尺寸，则不需要缩放
    sc = min(226 / w, 226 / h)
    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

img = (img/255.)*2 - 1

print(img.shape)