import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load Fashion MNIST dataset
(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Class names as per Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Directory to save images
save_dir = 'fashion_mnist_test_images'
os.makedirs(save_dir, exist_ok=True)

# Save first 10 test images
for i in range(10):
    img_array = x_test[i]
    label = y_test[i]
    class_name = class_names[label]
    
    img = Image.fromarray(img_array)
    filename = f'{save_dir}/{i}_{class_name}.png'
    img.save(filename)

print(f"Saved 10 Fashion MNIST test images to: {save_dir}")