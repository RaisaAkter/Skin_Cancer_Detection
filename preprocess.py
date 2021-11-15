''' 15-11-2021
Author: Raisa
This Code is for processing the images.
1. First the data directory have to be defined
2. Then for each image, first perform hair removal operation and save temporarily. 
    Then perform contrast enhancement and store the images according to class name. '''
from PIL import Image
from PIL import ImageEnhance
import cv2
#from cv2 import utils
import keras
#from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt 

def preprocessing():    
    data_dir = 'D:/CSE/MSc/1st Semester/CSE6265 Digital Image Processing/Research_related/Data/train/Melanoma'
    temporary = 'D:/CSE/MSc/1st Semester/CSE6265 Digital Image Processing/Research_related/temporary/'
    destination = 'D:/CSE/MSc/1st Semester/CSE6265 Digital Image Processing/Research_related/processed_img/melanoma/'
    images = sorted(os.listdir(data_dir),
            key = lambda x: str (x.split(".")[0]))

    images = [os.path.join(data_dir, img_path) 
                    for img_path in images]
    print(len(images))

    print("processing images ...")
    # loading image from absolute directory using opencv
    k=0
    for i, img_dir in enumerate(images):
        img = cv2.imread(img_dir)
        # Convert the original image to grayscale
        grayScale = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1,(17,17))

        # Perform the blackHat filtering on the grayscale image to find the 
        # hair countours
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        #cv2.imwrite('blackhat_sample1.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # intensify the hair countours in preparation for the inpainting algorithm
        ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
        
        # inpaint the original image depending on the mask
        dst = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA)
        cv2.imwrite(temporary+ 'melanoma'+ str(k) + '.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img = Image.open(temporary+ 'melanoma'+ str(k) + '.jpg')

        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.5
        img_contrasted = enh_con.enhance(contrast)
        img_contrasted.save(destination + 'melanoma'+ str(k) + '.jpg')
        k = k+1 
preprocessing()
print("Done Processing...")