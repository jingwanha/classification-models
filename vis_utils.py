# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.preprocessing import image
import keras.backend as K

import numpy as np
import cv2
import keras 

import tensorflow as tf
from tensorflow.python.framework import ops

def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)


    return x

class GradCAM:
    def __init__(self, model, activation_layer, class_idx):
        self.model = model
        self.activation_layer = activation_layer
        self.class_idx = class_idx
        self.tensor_function = self._get_gradcam_tensor_function()

    # get partial tensor graph of CNN model
    def _get_gradcam_tensor_function(self):
        model_input = self.model.input
        y_c = self.model.outputs[0].op.inputs[0][0, self.class_idx]
        
        try: A_k = self.model.get_layer(self.activation_layer).output
        except: 
            A_k = self.model.layers[0].get_layer(self.activation_layer).output
            y_c = self.model.layers[0].outputs[0].op.inputs[0][0, self.class_idx]

        tensor_function = K.function([model_input], [A_k, K.gradients(y_c, A_k)[0]])
        return tensor_function

    # generate Grad-CAM
    def generate(self, input_tensor):
        print (input_tensor)
        print (type(input_tensor))
        
        [conv_output, grad_val] = self.tensor_function([input_tensor])
        conv_output = conv_output[0]
        grad_val = grad_val[0]

        weights = np.mean(grad_val, axis=(0, 1))

        grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
        for k, w in enumerate(weights):
            grad_cam += w * conv_output[:, :, k]

        grad_cam = np.maximum(grad_cam, 0)

        return grad_cam, weights