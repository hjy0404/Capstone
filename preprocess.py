#unet output을 활용하여 원본이미지를 repvgg input에 맞게 preprocessing 후 저장
import glob
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import os
import matplotlib.pyplot as plt
def image_preprocessing(path):
    imglist = sorted(glob.glob(path+'/img/*/*.png'))
    npylist = sorted(glob.glob(path+'/npy/*/*.png'))
    for i in range(0,len(imglist)):
        img = (cv2.imread(imglist[i]))
        npimg = (cv2.imread(npylist[i]))
        npimg = np.where(npimg!=0,1,npimg)
        newimg = npimg * img
        pathname = os.path.dirname(imglist[i])
        print(os.path.basename(npylist[i]))
        if '1++' in pathname:
            cv2.imwrite(path+'/cow/preprocessed_image/1++/'+os.path.basename(os.path.splitext(npylist[i])[0])+'.jpg',newimg)
        elif '1+' in pathname:
            cv2.imwrite(path+'/cow/preprocessed_image/1+/'+os.path.basename(os.path.splitext(npylist[i])[0])+'.jpg',newimg)
        elif '1' in pathname:
            cv2.imwrite(path+'/cow/preprocessed_image/1/'+os.path.basename(os.path.splitext(npylist[i])[0])+'.jpg',newimg)
        elif '2' in pathname:
            cv2.imwrite(path+'/cow/preprocessed_image/2/'+os.path.basename(os.path.splitext(npylist[i])[0])+'.jpg',newimg)
        elif '3' in pathname:
            cv2.imwrite(path+'/cow/preprocessed_image/3/'+os.path.basename(os.path.splitext(npylist[i])[0])+'.jpg',newimg)
image_preprocessing("/content/drive/MyDrive/Colab Notebooks")
