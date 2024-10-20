user_input = st.text_input("Enter a sentence:", "What is the course fee")

if st.button("Predict"):
    text = user_input  # Use the user input as the initial text
    predictions = []  # To store predicted words

    # Iterate to predict the next 16 words
    for i in range(16):
        # Tokenize
        tokenize_text = tokenizer.texts_to_sequences([text])[0]

        # Padding
        text_padded = pad_sequences([tokenize_text], maxlen=56, padding='pre')

        # Prediction
        pos = np.argmax(model.predict(text_padded))

        # Get the predicted word
        for word, index in tokenizer.word_index.items():
            if index == pos:
                predicted_word = word
                break

        # Append the predicted word to the text
        text += " " + predicted_word
        predictions.append(predicted_word)  # Store the prediction for later use

        # Optional: Add a small delay to simulate a typing effect (comment out if not needed)
        # time.sleep(0.5)

    # Display the complete predicted sentence
    st.write("Predicted sentence:", text)