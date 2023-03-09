import time
import pickle
from subprocess import run
from random import randrange, choice
from pprint import pprint

import cv2
import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow as tf

load_path = 'fruit_model_save'
model = models.load_model(load_path)

with open(f'{load_path}/class_names.data', 'rb') as f:
    class_names = pickle.load(f)
print(class_names)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera.")
    quit()
else:
    print("Opened camera.")

time.sleep(1)

frames = 0
start = time.time()
while True:
    ret, img = cap.read()
    if ret: 
        frames += 1
    else:
        break
    new_img = cv2.resize(img, (258, 320))
    img_array = utils.img_to_array(new_img)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))
    fps = frames / (time.time() - start)
    cv2.putText(
        img, 
        f"{fps:0.0f} fps, {class_names[prediction]}", 
        (10,30), 
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, 
        (0, 255, 0),
        2
    )
    print(f"{frames:5} {class_names[prediction]:10}", end=" ")
    if frames % 5 == 0:
        print()
    cv2.imshow("Camera Feed", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print()
quit()