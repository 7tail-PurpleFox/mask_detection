from cProfile import label
import cv2
import os
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET
import numpy as np
import tqdm



prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"


# 下載模型相關檔案
if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
    urlretrieve(f"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/{prototxt}",
                prototxt)
    urlretrieve(f"https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/{caffemodel}",
                caffemodel)

net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

zero_face_files = ["NewFace Mask Dataset\Correct\Cotton\Child\FM 4169.jpg"]


for file_path in zero_face_files:
    image = cv2.imread(file_path)
    if image is None:
        # 讀取失敗則跳過
        continue
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # clip to image bounds
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)
        if endX <= startX or endY <= startY:
            continue
        faces.append({'box': (startX, startY, endX - startX, endY - startY), 'confidence': conf})
    if len(faces) == 0:
        print(f"No faces detected in {file_path} on re-check.")
    else:
        
        if len(faces) > 1:
            # Only keep the highest confidence face
            faces = [max(faces, key=lambda x: x['confidence'])]
        print(faces)
