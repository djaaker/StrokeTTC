#can probably ignore, made to convert a .pth file to a .ckpt

import torch
import tensorflow as tf
import numpy as np

#pt_file = "C:/Users/aprib/Downloads/best_model_weights.pth"

class YourTFModel(tf.keras.Model):
    def __init__(self):
        super(YourTFModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


# Load the PyTorch checkpoint
def load_pytorch_checkpoint(pytorch_path):
    checkpoint = torch.load(pytorch_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


# Map PyTorch weights to TensorFlow model
def map_weights(pytorch_weights, tf_model):
    for pt_name, pt_tensor in pytorch_weights.items():
        print(f"Processing PyTorch layer: {pt_name}")
        
        # Map PyTorch weight names to TensorFlow layer weights manually
        # This example assumes a simple fully connected layer mapping
        if 'dense1.weight' in pt_name:
            tf_model.dense1.kernel.assign(pt_tensor.numpy().T)  # Transpose weights for FC layers
        elif 'dense1.bias' in pt_name:
            tf_model.dense1.bias.assign(pt_tensor.numpy())
        elif 'dense2.weight' in pt_name:
            tf_model.dense2.kernel.assign(pt_tensor.numpy().T)  # Transpose weights
        elif 'dense2.bias' in pt_name:
            tf_model.dense2.bias.assign(pt_tensor.numpy())
        else:
            print(f"Skipping unknown layer: {pt_name}")


# Save the TensorFlow model as a checkpoint
def save_as_checkpoint(tf_model, ckpt_path):
    ckpt = tf.train.Checkpoint(model=tf_model)
    ckpt.write(ckpt_path)
    print(f"TensorFlow checkpoint saved to {ckpt_path}")


# Main conversion function
def convert_pytorch_to_tensorflow(pytorch_path, ckpt_path):
    # Load PyTorch weights
    pytorch_weights = load_pytorch_checkpoint(pytorch_path)

    # Define TensorFlow model
    tf_model = YourTFModel()

    # Map weights from PyTorch to TensorFlow
    map_weights(pytorch_weights, tf_model)

    # Save as TensorFlow checkpoint
    save_as_checkpoint(tf_model, ckpt_path)


# Path to PyTorch .pth file and output TensorFlow .ckpt file
pytorch_path = "C:/Users/aprib/Downloads/best_model_weights.pth"
ckpt_path = "path/to/your_model.ckpt"

# Convert the model
convert_pytorch_to_tensorflow(pytorch_path, ckpt_path)
