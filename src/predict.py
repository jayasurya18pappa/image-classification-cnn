import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/cnn_model.h5')

img = cv2.imread('data/test.jpg')
img = cv2.resize(img, (150,150))
img = img / 255.0
img = np.reshape(img, (1,150,150,3))

result = model.predict(img)

if result[0][0] > 0.5:
    print("Dog")
else:
    print("Cat")
