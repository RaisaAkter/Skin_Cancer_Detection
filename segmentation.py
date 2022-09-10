''' 12-06-2022
Author: Raisa
This Code is for Segmenting the images of 3 class.
For the segmentation purpose the contour based method is used '''

from PIL import Image
import cv2
import keras
import numpy as np
import os
import matplotlib.pyplot as plt 

def segmentation():    
    label='nevi'
    train_data_dir = 'D:/CSE/MSc/1st Semester/CSE6265 Digital Image Processing/Research_related/processed_img/train/'+label
    destination = 'D:/CSE/MSc/1st Semester/CSE6265 Digital Image Processing/Research_related/segmented_image/'+label+'/'
   
    train_images = sorted(os.listdir(train_data_dir),
            key = lambda x: str (x.split(".")[0]))

    train_images = [os.path.join(train_data_dir, img_path) 
                    for img_path in train_images]
    print(len(train_images))

    print("Segmenting images ...")
    
    # loading image from absolute directory using opencv
    k=0
    for i, img_dir in enumerate(train_images):
        img = cv2.imread(img_dir)
        img = cv2.resize(img,(256,256))
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((256,256), np.uint8)
        masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)
        segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        #plt.imshow(segmented)
        Image.fromarray(segmented).convert("RGB").save(destination + label+ str(k) + '.png')
        k = k+1 
segmentation()
print("Done Segmentation...")