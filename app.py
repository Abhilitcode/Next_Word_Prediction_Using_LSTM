import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 

#load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

#load the model
model = load_model('lstm_model.keras')

#give some title
st.title('Text Prediction with LSTM')

#take the user input
user_input = st.text_input('Enter the sentence:', "What is the course fee?")

#prediction
if st.button("Predict"):
    text = user_input
    predictions = []
    
    #iterate
    for i in range(16):
        tokenize_text = tokenizer.texts_to_sequences([text])[0]
        padded_text = pad_sequences([tokenize_text], maxlen = 56, padding='pre')
        
        #predict
        pos = np.argmax(model.predict(padded_text))
        
        for word, index in tokenizer.word_index.items():
            if index == pos:
                predicted_word = word
                break
        
        text += " "+ predicted_word
        predictions.append(predicted_word)
    st.write("Predicted sentence is:", text)

