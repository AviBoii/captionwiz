import numpy as np
import pickle
import streamlit as st
import tensorflow
from PIL import Image
import pandas as pd
import cv2
import io

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer

loaded_captioning_model=pickle.load(open("/Users/amanrev/Documents/IET/captionwiz/trained_captioning_model.sav",'rb'))

inception_v3 = InceptionV3()
inception_v3 = Model(inputs = inception_v3.inputs, outputs = inception_v3.layers[-2].output)
inception_v3.summary()

tokenizer = Tokenizer()
# Function to preprocess the image

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = '<start>'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text



def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# def load_img(img_path):
#     img = tensorflow.io.read_file(img_path)
#     img = tensorflow.io.decode_jpeg(img, channels=3)
#     img = tensorflow.keras.layers.Resizing(299, 299)(img)
#     img = img/255
#     return img 

def get_feature_vector(img):
    img = tensorflow.expand_dims(img, axis=0)
    feature_vector = inception_v3(img)
    feature_vector = np.array(feature_vector)
    print(feature_vector)
    return feature_vector



# Function to generate caption
def generate_caption(image):
    y_pred = predict_caption(loaded_captioning_model, get_feature_vector(image), tokenizer, 35) 
    return y_pred  # Replace this with the actual generated caption

# Main function to define the Streamlit app
def main():
    st.title("CaptionWiz: Image Captioning")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        img = tensorflow.io.decode_jpeg(image_bytes, channels=3)
        img = tensorflow.keras.layers.Resizing(299, 299)(img)
        img = img/255
        # Display the uploaded image
    #     image = Image.open(uploaded_file)
    
    # # Resize the image to 299x299
    #     resized_image = image.resize((299, 299))
    
    # # Convert PIL Image to TensorFlow tensor
    #     numpy_image = np.array(resized_image)

    
    #     numpy_image = numpy_image.astype(np.float32) / 255.0
    
    # Display the resized image
        # st.image(img, caption="Resized Image (299x299)", use_column_width=True)
        # Button to generate caption
        if st.button("Generate Caption"):
            # Generate caption and display
            with st.spinner('Generating caption...'):
                caption = generate_caption(img)
                st.success("Caption: {}".format(caption))

# Run the app
if __name__ == "__main__":
    main()
