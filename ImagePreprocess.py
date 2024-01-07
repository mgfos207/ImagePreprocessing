import numpy as np

class ImagePreprocess(object):
    def __init__(self, destination_dir:str, img_dir:str, img_dims:tuple=(200, 200)) -> None:
        self.img_dir = img_dir
        self.dest_dir = destination_dir
        self.img_dims = img_dims
    
    def preprocess_img(self, cv2, anime_face_detection, idx):

        img = cv2.imread(f"{self.img_dir}")
        height, width, ch = img.shape
        min_width = width * .1
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = anime_face_detection.detectMultiScale(img_gray, 
                            minSize =(20, 20))
        amount_found = len(faces)

        if amount_found != 0:
            count = 0
            for x, y, w, h in faces:
                left = x
                top = y
                right = x + w
                width = right - left
                bottom = y + h
                if width >= min_width:
                    face_img = img[int(top):int(bottom), int(left):int(right)] # crop
                    resize_img = cv2.resize(face_img, (250, 250))
                    cv2.imwrite(f"{self.dest_dir}/{idx}_{count}.jpg",resize_img)
                count += 1
