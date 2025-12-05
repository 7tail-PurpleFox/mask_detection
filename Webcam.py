import argparse
import time
import cv2
import numpy as np
from retinaface import RetinaFace
import tensorflow as tf
from tensorflow import keras
from imutils.video import WebcamVideoStream

# ---------------------------------------------------------
# argparse
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="mask_detector_mobilenet.h5")
    parser.add_argument("--size", type=int, default=224)
    return vars(parser.parse_args())

args = parse_args()
size = args["size"]

# ---------------------------------------------------------
# Load mask classifier
# ---------------------------------------------------------
model = tf.keras.models.load_model(args["model"], compile=False)
data = np.ndarray(shape=(1, size, size, 3), dtype=np.float32)

mask_label = {0: 'MASK INCORRECT', 1: 'MASK', 2: 'NO MASK'}
color = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255)}

# ---------------------------------------------------------
# RetinaFace detection function
# ---------------------------------------------------------
def detect_faces(img):
    faces = RetinaFace.detect_faces(img)
    rects = []
    if isinstance(faces, dict):
        for key in faces.keys():
            identity = faces[key]
            x1, y1, x2, y2 = identity["facial_area"]
            confidence = identity.get("score", 1.0)
            rects.append({"box": (x1, y1, x2-x1, y2-y1), "confidence": confidence})
    return rects

# ---------------------------------------------------------
# Webcam
# ---------------------------------------------------------
vs = WebcamVideoStream().start()
time.sleep(1.0)
start = time.time()
count = 0
while True:
    count += 1
    frame = vs.read()
    if count % 5 == 0:
        rects = detect_faces(frame)
    else:
        continue

    # -------------------------------
    # Process each face
    # -------------------------------
    for rect in rects:
        (x, y, w, h) = rect['box']
        dx = int(x-w*0.2)        #擴大臉的範圍
        if dx<0:
            dx = 0
        dy = int(y-h*0.2)
        if dy<0:
            dy = 0
        dw = int(w*1.4)
        if dx+dw>frame.shape[1]:
            dw = frame.shape[1]-dx
        dh = int(h*1.4)
        if dy+dh>frame.shape[0]:
            dh = frame.shape[0]-dy
        face_img = frame[dy:dy+dh, dx:dx+dw]

        if face_img.size == 0:
            continue

        # MobileNet preprocess
        face_img = cv2.resize(face_img, (size, size))
        face_img = keras.preprocessing.image.img_to_array(face_img)
        face_img = keras.applications.mobilenet_v2.preprocess_input(face_img)

        data[0] = face_img
        prediction = model.predict(data, verbose=0)
        result = np.argmax(prediction[0])
        confidence = prediction[0][result]
        if confidence < args["confidence"]:
            continue

        # draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color[result], 2)
        label = f"{mask_label[result]} {confidence*100:.2f}%"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[result], 2)

    # FPS
    end = time.time()
    fps = 1 / (end - start)
    start = end
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

