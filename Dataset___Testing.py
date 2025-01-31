

import os
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from Dataset_CD import change_detection_dataset
from Model_gpu import UNETR_2D_CD
from Loss import DiceLoss, DiceBCELoss
from Utils import seeding

""" Dataset """
patch_size = 16
train_path="D://Datasets//Levir_croped_256//LEVIR_CD//train"
val_path="D://Datasets//Levir_croped_256//LEVIR_CD//val"
test_path="D://Datasets//Levir_croped_256//LEVIR_CD//test"

""" Dataloader """
batch_size = 8

train_dataset = change_detection_dataset(root_path=train_path, patch_size=patch_size)
val_dataset = change_detection_dataset(root_path=val_path, patch_size=patch_size)
test_dataset = change_detection_dataset(root_path=test_path, patch_size=patch_size)


train_loader=DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=False)
val_loader=DataLoader(val_dataset, batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=False)
test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=False)


for x, y, l, f in train_loader:
        print (x.shape)     ## 8, 256, 768      --> we already put batch=8
        print (y.shape)     ## 8, 256, 256      --> we already put batch=8
        print (l.shape)     ## 8, 1, 256, 256   --> we already put batch=8
        print (f)                               ## print names of the files
        print (x.device)
        break

x1 = x[0]
print (x1.shape)
y1 = y[0]
print (y1.shape)
l1 = l[0]
print (l1.shape)

for i in range(16):
  for j in range(16):
    plt.subplot(16,16,i*16+j+1)
    plt.imshow(x1[i*16+j].reshape(3, 16, 16).permute(1, 2, 0).numpy())
    plt.axis('off')

plt.show()
for i in range(16):
  for j in range(16):
    plt.subplot(16,16,i*16+j+1)
    plt.imshow(y1[i*16+j].reshape(3, 16, 16).permute(1, 2, 0).numpy())
    plt.axis('off')

plt.show()
plt.imshow(l1.permute(1, 2, 0).numpy())
plt.axis('off')

plt.show()
