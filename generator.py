import tensorflow as tf
import numpy as np
from PIL import Image

@tf.keras.utils.register_keras_serializable()
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

# Function to preprocess the input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize to match the model's input size
    img = np.array(img).astype(np.float32)
    img = (img / 127.5) - 1  # Normalize to [-1, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to postprocess the output image
def postprocess_image(img):
    img = (img * 0.5 + 0.5) * 255  # Denormalize to [0, 255]
    return img.astype(np.uint8)

# Load the saved generator model with custom objects
custom_objects = {'InstanceNormalization': InstanceNormalization}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('generator_g.h5', compile=False)

# Load and preprocess the input image
input_image = preprocess_image('input.jpg')

# Generate the output image
output_image = model.predict(input_image)

# Postprocess and save the output image
output_image = postprocess_image(output_image[0])
Image.fromarray(output_image).save('output.png')

print("Image processing complete. Output saved as 'output.jpg'.")