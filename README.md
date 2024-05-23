# Character Level Language Model using Shakespeare Dataset

This project aims to generate new sentences by learning character-level sequences using RNNs. The dataset is a collection of Shakespeare's works, utilized for training the model. This project is part of the 'Artificial Neural Networks and Deep Learning' class in the Spring Semester of Data Science at Seoul National University of Science and Technology in 2024.

## Character Level Language Model
The language model is designed to predict the next character in a sequence, enabling the generation of text character by character.

## Dataset
The dataset consists of texts from Shakespeare's novels, preprocessed to create sequences of characters for model training.

## Software Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Key Files
- `dataset.py`: Contains the data pipeline for preprocessing the Shakespeare dataset.
- `model.py`: Implements the RNN and LSTM models.
- `main.py`: Script to train the models and track loss.
- `generate.py`: Script to generate text using the trained models.

## Usage
1. **Prepare the Dataset**: Ensure the Shakespeare text file is available and correctly referenced in `dataset.py`.
2. **Train the Model**: Run `main.py` to train both RNN and LSTM models.
3. **Generate Text**: Use `generate.py` to generate text from the trained models.

## Installation
To install the required libraries, run:
```sh
pip install torch numpy matplotlib
