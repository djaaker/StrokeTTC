import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import django
from django.conf import settings
import tensorflow as tf
from tensorflow.python.training import checkpoint_utils
import torch
import sys
matplotlib.use('Agg')  # Use non-GUI backend

# Add the parent directory of 'myproject' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the Django settings module environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')  # Replace 'myproject' with your Django project name

# Setup Django
django.setup()

# file_path = "C:/Users/aprib/Downloads/best_model_weights.pth"
# data = torch.load(file_path)

# # Extract data for plotting (modify this based on your .pth file structure)
# # Example: Assuming 'data' contains a dictionary of keys and values
# keys = list(data.keys())
# values = list(data.values())
#print(values)
# confidence_intervals = [2,2,2]

yeahokay = "C:/Users/aprib/Downloads/best_model_weights.pth" #put your path here 

def create_bar_graph_with_ci_from_pth(pth_file_path):
    """
    Creates a bar graph with confidence intervals using data from a .pth file.

    Args:
    - pth_file_path (str): Path to the .pth file.
    """
    # Load the .pth file
    try:
        data = torch.load(pth_file_path)
    except Exception as e:
        raise ValueError(f"Error loading .pth file: {e}")

    weights = [value.mean().item() for key, value in data.items() if "weight" in key]
    biases = [value.mean().item() for key, value in data.items() if "bias" in key]
    categories = [key for key in data.keys() if "weight" in key]

    # Ensure weights and biases are aligned
    if len(weights) != len(biases):
        raise ValueError("Mismatch between number of weights and biases.")

    # Create confidence intervals (e.g., standard deviation or some fixed values)
    confidence_intervals = [abs(bias) for bias in biases]
    #print(biases)
    # Extract keys, values, and confidence intervals from the .pth file
    # try:
    #     keys = list(data.keys())  # Assuming `keys` is stored as a list of category names
    #     #print(keys)
    #     values = list(data.values())  # Assuming `values` is stored as a list of bar heights
    #     confidence_intervals = data.get('confidence_intervals')  # Assuming CI values are stored
        
    #     if not (keys and values and confidence_intervals):
    #         raise ValueError("Missing 'keys', 'values', or 'confidence_intervals' in .pth file.")
    #     if len(keys) != len(values) or len(keys) != len(confidence_intervals):
    #         raise ValueError("Length of keys, values, and confidence intervals must match.")
    # except KeyError as e:
    #     raise ValueError(f"Expected data missing in .pth file: {e}")

    # Clear any previous plots
    plt.clf()
    plt.close('all')

    # Create the bar graph
    fig, ax = plt.subplots()

    x_positions = range(len(categories))
    bars = ax.bar(x_positions, weights, yerr=confidence_intervals, capsize=5, label="Value")

    # Label each bar with its value as a percentage
    for bar, value, ci in zip(bars, weights, confidence_intervals):
        y_position = bar.get_height() + ci + 1  # Adjust position above the bar + CI
        ax.text(bar.get_x() + bar.get_width() / 2, y_position, f"{value}% ± {ci}%", 
                ha='center', va='bottom', fontsize=10, color='black')

    # Labels and title
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=0, ha='right')
    ax.set_xlabel('Potential Diagnoses')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Diagnosis Graph with Confidence Intervals')

    # Save the plot as an image in the Django static folder
    output_path = os.path.join(settings.BASE_DIR, 'static', 'bar_graph_ci22.png')
    print(f"Saving image to: {output_path}")
    plt.tight_layout()
    plt.savefig(output_path)  # Save to Django static folder
    plt.close()

create_bar_graph_with_ci_from_pth(yeahokay)