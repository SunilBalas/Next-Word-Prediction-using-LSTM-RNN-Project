import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Prediction:
    def __init__(self):
        pass
    
    def predict_next_word(self, model, tokenizer, text, max_sequence_len, next_word_len=1):
        try:
            token_list = tokenizer.texts_to_sequences([text])[0]
            if len(token_list) >= max_sequence_len:
                token_list = token_list[-(max_sequence_len - 1):] # Ensure the sequence length matches max_sequence_len - 1
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            
            predicted = model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted, axis=1)
            
            pred_next_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    pred_next_word = word
                    break
            text += " " + pred_next_word
            
            if next_word_len > 1:
                return self.predict_next_word(model, tokenizer, text, max_sequence_len, next_word_len - 1)
            return text
            
        except Exception as ex:
            raise ex