from glob import glob
import os
import random
from shutil import copyfile

random.seed(42)

training_image_dirs = glob('./ImageNet/ILSVRC2012_img_train/*')
imgs = [glob(f'{dir}/*') for dir in training_image_dirs]
imgs = [img for img_dir in imgs for img in img_dir]


for i in range(50):

    random_sample = random.sample(imgs, 500)
    os.makedirs(f'./ImageNet/random500_{i}', exist_ok=True)
    
    for img in random_sample:

        img_file = img.split('/')[-1]
        copyfile(img, f'./ImageNet/random500_{i}/{img_file}')