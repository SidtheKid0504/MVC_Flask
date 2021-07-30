import numpy as np
from numpy.core.fromnumeric import clip
import pandas as pd
import PIL.ImageOps as pio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from PIL import Image

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

trainX, testX, trainy, testy = train_test_split(X, y, random_state=9, train_size = 7500, test_size=2500)

trainX = trainX / 255
testX = testX / 255

lr = LogisticRegression(solver='saga', multi_class='multinomial')
lr.fit(trainX, trainy)

ypreds = lr.predict(testX)
acc = accuracy_score(testy, ypreds)

print(str(acc*100) + '%')

def predict_input_image(img):
    conv_img = Image.open(img).convert('L')
    resized_img = conv_img.resize((28, 28), Image.ANTIALIAS)

    pixel_filter = 20
    min_pixel = np.percentile(resized_img, pixel_filter)
    max_pixel = np.max(resized_img)

    scaled_img = np.clip(resized_img - min_pixel, 0, 255)
    scaled_img = np.asarray(scaled_img) / max_pixel

    test_sample = np.asarray(scaled_img).reshape(1, 784)
    test_pred = lr.predict(test_sample)

    return test_pred[0]