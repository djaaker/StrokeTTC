#can probably ignore, made to convert a .pt file to a .ckpt

import torch
import tensorflow as tf
import numpy as np

pt_file = "C:/Users/aprib/Downloads/psp_celebs_seg_to_face.pt/psp_celebs_seg_to_face.pt"
checkpoint = torch.load(pt_file)

# If the checkpoint contains additional metadata, extract the model's state_dict
model_state_dict = checkpoint.get('model_state_dict', checkpoint)

class YourModel(tf.keras.Model):
    def __init__(self):
        super(YourModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

tf_model = YourModel()

# Example of mapping weights
for pt_name, pt_weight in model_state_dict.items():
    print(f"Loading PyTorch layer: {pt_name}")
    tf_layer = ...  # Find corresponding TensorFlow layer based on `pt_name`
    
    # Convert PyTorch weights to NumPy and assign to TensorFlow
    if 'weight' in pt_name:
        tf_layer.kernel.assign(np.transpose(pt_weight.numpy()))
    elif 'bias' in pt_name:
        tf_layer.bias.assign(pt_weight.numpy())

ckpt = tf.train.Checkpoint(model=tf_model)

# Save the checkpoint
ckpt_path = "C:/Users/aprib/Downloads/psp_celebs_seg_to_face.pt/psp_celebs_seg_to_face.ckpt"
ckpt.write(ckpt_path)

print(f"TensorFlow checkpoint saved to {ckpt_path}")