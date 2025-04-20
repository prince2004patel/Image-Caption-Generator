from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

resnet_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(img_path):
    img_array = preprocess_image(img_path)
    features = resnet_model.predict(img_array)
    return features
