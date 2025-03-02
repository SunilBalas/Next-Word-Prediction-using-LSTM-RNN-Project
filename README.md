# Next Word Prediction using LSTM RNN Project

This project implements a Next Word Prediction system using Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN). The model is trained on text data to predict the next word given an input sequence. A Streamlit-based UI is developed for easy interaction with the model.

## Features

  - **Data Collection**: Gathers textual data from various sources for training.
  - **Data Preprocessing**: Cleans, tokenizes, and converts text into sequences suitable for training.
  - **Model Building and Training**: Implements an LSTM-based neural network for next-word prediction.
  - **Model Evaluation**: Assesses model performance using standard evaluation metrics.
  - **User Interface**: A Streamlit-powered UI allows users to interact with the trained model and generate predictions in real-time.

## Installation and Setup

  1. Clone the repository
  ```
  git clone https://github.com/SunilBalas/Next-Word-Prediction-using-LSTM-RNN-Project.git
  ```

  2. Install dependencies
  ```python
  pip install -r requirements.txt
  ```

  3. Usage

  ```python
  streamlit run app.py
  ```

## Project Structure

```
Next-Word-Prediction-using-LSTM-RNN-Project/
│── data/               
    |── models/                # Store the models and tokenizers
        │── model.h5
        │── tokenizer.pickle
    |── raw/                   # Raw form dataset storage
    |── processed/      
        |── train/             # Store the train set of dataset
            │── X_train.csv
            │── y_train.csv
        |── test/              # Store the test set of dataset
            │── X_test.csv
            │── y_test.csv
│── notebooks/                 # Jupyter experiment notebooks for analysis
│── src/                       # Source code
    │── components/
        │── artifacts.py       # Source code for the load model and tokenizer
        │── prediction.py      # Souce code for predicting next word method
     ├── app.py                # Streamlit app for UI
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
```

## Model Details

  - Architecture: Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN)
  - Framework: TensorFlow/Keras
  - Training Data: Text corpus
  - Optimization: Adam optimizer with categorical cross-entropy loss
  - Future Enhancements: Integrate attention mechanisms for better predictions.

## License

  - This project is licensed under the GNU General Public License v3.0.

## Contribution

1. Fork the repository
2. Create your feature branch: 
```
git checkout -b my-new-feature
```
3. Commit your changes: 
```
git commit -a -m 'Add some feature'
```
4. Push to the branch: 
```
git push origin my-new-feature
```
5. Submit a pull request

## Acknowledgments

  - The developers who created the open-source libraries and tools used in this project, anyone who has contributed to the project.
  - Special thanks to [Krish Naik](https://github.com/krishnaik06) for explaining complex Neural Network architectures so easily through their Udemy courses.
