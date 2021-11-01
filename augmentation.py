from keras.preprocessing.image import ImageDataGenerator
from skimage import io
datagen = ImageDataGenerator(        
        rotation_range = 90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range = (0.5, 1.5))
import numpy as np
import os
from PIL import Image
image_directory = r'D:/CSE/MSc/1st Semester/CSE6265 Digital Image Processing/Research_related/Data/train/Kert/'
SIZE = 224
dataset = []
my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):    
    if (image_name.split('.')[1] == 'jpg'):        
        image = io.imread(image_directory + image_name)        
        image = Image.fromarray(image, 'RGB')        
        image = image.resize((SIZE,SIZE)) 
        dataset.append(np.array(image))
x = np.array(dataset)
i = 0
for batch in datagen.flow(x, batch_size=756,
                          save_to_dir= r'D:/CSE/MSc/1st Semester/CSE6265 Digital Image Processing/Research_related/Data/train/Augmented-images',
                          save_prefix='kert',
                          save_format='jpg'):    
    i += 1    
    if i > 4:        
        break