import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
class_names = ["car", "human"]


def prediction(bImage):
    try:
        model = keras.models.load_model(
            'static/cnp_model')
        print("model success")

        image = tf.io.decode_image(
            bImage, dtype=tf.float32)

        image = image[tf.newaxis, ...]

        prediction = model.predict(image)

        return class_names[np.argmax(prediction)]
    except Exception as e:
        print(e)
