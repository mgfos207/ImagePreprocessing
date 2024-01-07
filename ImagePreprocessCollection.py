from ImagePreprocess import ImagePreprocess
import os
import shutil
import cv2
import splitfolders
import numpy as np
import math

class ImagePreProcessCollection:
    def __init__(self, image_directory:str, split:tuple, destination_dir:str, split_dir:str, img_dims:tuple=(200, 200)):
        """
        what do we need when doing image precessing?
        1) Current image directory
        2) How we want to split the data (train/test/val)
        image directory
        3) Destination of where we will host the split files
        4) What is the image size we need to resize the data to prior to feeding into a neural network
        """
        self.img_dir = image_directory
        self.dest_dir = destination_dir
        self.split_dir = split_dir
        self.split = split
        self.img_dims = img_dims
        self.anime_face_detection = cv2.CascadeClassifier('lbpcascade_animeface.xml')
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        if not os.path.exists(self.split_dir):
            os.makedirs(self.split_dir)

    def preprocess_imgs(self):
        for dir, path, files in os.walk(self.img_dir):
            for idx,file in enumerate(files):
                allowed_exts = set(['jpg', 'png', 'jpeg'])
                file_ext = file.split(".")[-1]
                if file_ext in allowed_exts:
                    img_file = f"{self.img_dir}/{file}"
                    try:
                        img_pre = ImagePreprocess(self.dest_dir, img_file, self.img_dims)
                        img_pre.preprocess_img(cv2, self.anime_face_detection, idx)
                    except Exception as e:
                        print(f"Issue with the following file {file}")
            break
    
    def split_dataset(self):
        """
        get length of files in dataset
        get train size in files
        get test size in files
        get validation size in files (if provided)
        """
        files = os.listdir(self.dest_dir)
        num_files = len(files)
        np.random.shuffle(files)
        if len(self.split) > 2:
            train_perc, test_perc, val_perc = self.split
            train_size = math.floor(num_files * train_perc)
            test_size = math.floor(num_files * test_perc)
            val_size = math.floor(num_files * val_perc)
            train_files = files[0:train_size]
            test_files = files[train_size: train_size + test_size]
            val_files = files[train_size + test_size:-1]

            #train files move
            if not os.path.exists(f"{self.split_dir}/train"):
                os.makedirs(f"{self.split_dir}/train")
            if not os.path.exists(f"{self.split_dir}/test"):
                os.makedirs(f"{self.split_dir}/test")
            if not os.path.exists(f"{self.split_dir}/val"):
                os.makedirs(f"{self.split_dir}/val")
        
            for file in train_files:
                shutil.move(f"{self.dest_dir}/{file}", f"{self.split_dir}/train")

            for file in test_files:
                shutil.move(f"{self.dest_dir}/{file}", f"{self.split_dir}/test")
                            
            for file in val_files:
                shutil.move(f"{self.dest_dir}/{file}", f"{self.split_dir}/val")

        elif len(self.split == 2):
            train_perc, test_perc = self.split
            train_size = math.floor(num_files * train_perc)
            test_size = math.floor(num_files * test_perc)