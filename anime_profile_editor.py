import os
import cv2
import numpy as np
import anime_face_detector import create_detector

pathRaw = 'raw_images/'
pathEdit = 'edited_images/'

imgList = os.listdir(pathRaw)

detector = create_detector('yolov3')
img = cv2.imread(pathRaw+imgList[0])
preds = detector(img)
print(preds)
