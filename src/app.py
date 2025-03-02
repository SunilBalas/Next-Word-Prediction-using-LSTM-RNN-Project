import streamlit as st
from components.artifacts import Artifacts
from components.prediction import Prediction

class App:
    def __init__(self):
        self.artifacts = Artifacts()
        self.prediction = Prediction()
    
    # Streamlit App
    def create_app(self):
        st.title("Next Word Prediction using LSTM RNN")
        input_text = st.text_area("Enter the sequence of words: ")
        
        next_word_len = st.number_input('Enter the number of predicted next words: ', min_value=1, max_value=50, step=1)

        # Load the LSTM model and tokenizer
        model = self.artifacts.load_model()
        tokenizer = self.artifacts.load_tokenizer()

        if st.button("Predict Next Word"):
            max_sequence_len = model.input_shape[1] + 1
            next_word = self.prediction.predict_next_word(model, tokenizer, input_text, max_sequence_len, next_word_len)
            st.write(f"Next Word: {next_word}")

if __name__ == "__main__":
    app = App()
    app.create_app()