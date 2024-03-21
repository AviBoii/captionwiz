# Captionwiz

## Overview 
Image captioning is a challenging task in the field of computer vision and natural language processing. The goal of this project is to automatically generate captions that accurately describe the content of an image.

## Technology used
python,pandas,numpy,tensorflow,keras,streamlit.

## Features
- ###  CNN Feature Extraction
   Utilizes the Inception model to extract high-level features from input images.
- ### LSTM Caption Generation:
    Employs a Long Short-Term Memory (LSTM) network to generate captions based on the extracted image features.
- ###  Streamlit Interface:
    Hosted using Streamlit, providing a user-friendly interface for uploading images and viewing generated captions.
- ### Pretrained Models:
   Includes pretrained weights for both the CNN and LSTM models to facilitate quick deployment and usage.
- ### Trained on Flickr Dataset:
   The models are trained on the Flickr dataset, a widely used benchmark dataset for image captioning tasks.

## Model Architecture

#### The architecture of the image captioning model consists of two main components:

- CNN (Convolutional Neural Network): Extracts high-level features from input images using the Inception model.

- LSTM (Long Short-Term Memory): Generates descriptive captions based on the features extracted by the CNN.

## Installation
- Cloning the repo
  git clone https://github.com/AMANREVANKAR/captionwiz.git
- Activating the virtual environment
   source env/bin/activate
- Runining the streamlit file
    streamlit run app.py

## Working
<img width="880" alt="Screenshot 2024-03-21 at 9 26 20 PM" src="https://github.com/AMANREVANKAR/captionwiz/assets/122635887/83e2b474-71a0-4f6d-bedb-710f6a6a517a">
