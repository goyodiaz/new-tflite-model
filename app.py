import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Arthopoda', 'Ascomycota', 'Bryophyta', 'Rhodophyta']

# Function to convert DNA sequence to color mapping
def dna_to_color(dna_sequence):
    """
    Convert a DNA sequence to a list of colors.
    A -> red, T -> green, G -> blue, C -> yellow
    Any unexpected character -> black
    """
    color_map = {'A': 'red', 'T': 'green', 'G': 'blue', 'C': 'yellow', '-': 'black', 'Y': 'black', 'N': 'black', 'W': 'black'}
    return [color_map.get(base, 'black') for base in dna_sequence]

# Function to generate barcode from DNA sequence
def generate_barcode(dna_sequence, output_path, image_width=512, image_height=265):
    """Generate a barcode image from a DNA sequence and save it."""
    colors = dna_to_color(dna_sequence)
    num_bars = len(dna_sequence)
    bar_width = image_width / num_bars
    bar_height = image_height

    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
    ax.axis('off')

    for idx, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((idx * bar_width, 0), bar_width, bar_height, color=color))

    ax.set_xlim(0, image_width)
    ax.set_ylim(0, bar_height)
    ax.set_aspect('auto')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100, transparent=True)
    plt.close()

    return output_path

# Function to generate overlapping codons from DNA sequence
def get_overlapping_codons(dna_seq):
    return [dna_seq[i:i+3] for i in range(len(dna_seq) - 2)]  # Overlapping codons

# Function to assign highly contrasting and bright colors to each codon
def get_codon_colors(codons):
    # Predefined color map for each unique codon with bright and contrasting colors
    codon_color_map = {
        'AAA': '#FF0000', 'AAT': '#00FF00', 'AAG': '#0000FF', 'AAC': '#FFFF00',
        'ATA': '#FF00FF', 'ATT': '#00FFFF', 'ATG': '#000000', 'ATC': '#FFFFFF',
        'AGA': '#800000', 'AGT': '#008000', 'AGG': '#000080', 'AGC': '#808000',
        'ACA': '#800080', 'ACT': '#008080', 'ACG': '#808080', 'ACC': '#FFC0CB',
        'TAA': '#FFA500', 'TAT': '#800080', 'TAG': '#FFFFE0', 'TAC': '#4682B4',
        'TTA': '#D2691E', 'TTT': '#FF1493', 'TTG': '#4B0082', 'TTC': '#FFD700',
        'TGA': '#ADFF2F', 'TGT': '#00CED1', 'TGG': '#FF4500', 'TGC': '#DA70D6',
        'TCA': '#EE82EE', 'TCT': '#9400D3', 'TCG': '#7FFF00', 'TCC': '#DC143C',
        'GAA': '#32CD32', 'GAT': '#FF6347', 'GAG': '#00FA9A', 'GAC': '#1E90FF',
        'GTA': '#8B4513', 'GTT': '#B22222', 'GTG': '#228B22', 'GTC': '#4169E1',
        'GGA': '#5F9EA0', 'GGT': '#FF8C00', 'GGG': '#8A2BE2', 'GGC': '#FF00FF',
        'GCA': '#8B008B', 'GCT': '#32CD32', 'GCG': '#00FFFF', 'GCC': '#DAA520',
        'CAA': '#CD5C5C', 'CAT': '#FFD700', 'CAG': '#696969', 'CAC': '#8FBC8F',
        'CTA': '#B0E0E6', 'CTT': '#DC143C', 'CTG': '#FF00FF', 'CTC': '#7CFC00',
        'CGA': '#4682B4', 'CGT': '#00FFFF', 'CGG': '#FF1493', 'CGC': '#FFD700',
        'CCA': '#DDA0DD', 'CCT': '#FF6347', 'CCG': '#FF4500', 'CCC': '#ADFF2F',
    }

    # Assign colors based on the predefined map
    colors = [codon_color_map.get(codon, 'black') for codon in codons]
    return colors

# Function to generate barcode from codons
def plot_codons_as_barcode(codons, output_path, image_width=512, image_height=265):
    """Generate a barcode image from a list of codons and save it."""
    colors = get_codon_colors(codons)  # Get the colors for each codon
    num_bars = len(codons)
    bar_width = image_width / num_bars
    bar_height = image_height

    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
    ax.axis('off')

    for idx, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((idx * bar_width, 0), bar_width, bar_height, color=color))

    ax.set_xlim(0, image_width)
    ax.set_ylim(0, bar_height)
    ax.set_aspect('auto')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100, transparent=True)
    plt.close()

    return output_path

# Function to prepare the image for prediction using TensorFlow Lite
def prepare_image(image_path, target_size=(512, 265)):
    """Prepare an image for prediction."""
    img = Image.open(image_path)

    # Ensure the image is in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image
    img = img.resize(target_size)

    # Convert to numpy array and normalize
    img = np.array(img) / 255.0

    # Convert to FLOAT32 type (important for TensorFlow Lite)
    img = img.astype(np.float32)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app
def run_streamlit_app():
    st.title("DNA Barcode Generator")

    # Input DNA sequence
    dna_sequence = st.text_area("Enter DNA sequence (without spaces or special characters):")

    if dna_sequence:
        # Generate barcode based on codon color barcode method
        output_file = "dna_barcode.png"
        
        # Generate codons and use the codon color mapping
        codons = get_overlapping_codons(dna_sequence)
        plot_codons_as_barcode(codons, output_file)

        # Display the result
        st.image(output_file, caption="Generated DNA Barcode", use_column_width=True)

        # Provide download link
        with open(output_file, "rb") as f:
            st.download_button("Download Barcode Image", f, file_name="dna_barcode.png")

if __name__ == "__main__":
    run_streamlit_app()

# Streamlit app for uploading the barcode image for prediction
st.title("DNA Sequence to Organism Predictor")

# Step 2: Upload the barcode image for prediction
st.header("Step 2: Upload Barcode Image for Prediction")
uploaded_file = st.file_uploader("Upload the barcode image (PNG):", type="png")
if uploaded_file:
    image_path = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Prepare the image and make prediction using TensorFlow Lite
    try:
        prepared_image = prepare_image(image_path)

        # Set the input tensor to the prepared image
        input_tensor_index = input_details[0]['index']
        interpreter.set_tensor(input_tensor_index, prepared_image)

        # Run inference
        interpreter.invoke()

        # Get prediction results
        output_tensor_index = output_details[0]['index']
        prediction = interpreter.get_tensor(output_tensor_index)

        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"Predicted Class: {predicted_class}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
