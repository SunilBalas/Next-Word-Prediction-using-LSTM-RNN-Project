import pickle
from pathlib import Path
from tensorflow.keras.models import load_model

class Artifacts:
    def __init__(self):
        # Set the absolute path to 'data/models/'
        self.dir_path = Path(__file__).resolve().parents[2] / 'data' / 'models'

    def load_model(self, model='lstm'):
        try:
            # Construct the model path
            model_path = self.dir_path / 'model.keras'
            model = load_model(model_path)
            return model
        except Exception as ex:
            raise Exception(f"Error loading the model '{model}': {str(ex)}")
        
    def load_tokenizer(self):
        try:
            # Construct the tokenizer path
            tokenizer_path = self.dir_path / 'tokenizer.pickle'
            with open(tokenizer_path, 'rb') as file:
                tokenizer = pickle.load(file)
            return tokenizer
        except Exception as ex:
            raise Exception(f"Error loading the tokenizer: {str(ex)}")
