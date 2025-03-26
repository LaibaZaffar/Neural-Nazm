# LSTM Poetry Generation in Roman Urdu

## Project Overview
This project implements a Long Short-Term Memory (LSTM) neural network for generating poetry in Roman Urdu. The model learns from a dataset of poetry and can generate new, creative text based on a seed phrase.

## Features
- Text preprocessing and cleaning
- LSTM-based sequence generation
- Temperature-based sampling for creative text generation
- Uses Roman Urdu poetry dataset

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - tensorflow
  - scikit-learn
  - re (regular expressions)

## Key Components

### Data Preprocessing
- Loads poetry from CSV file
- Cleans text by:
  - Converting to lowercase
  - Removing non-Roman Urdu characters
  - Stripping extra whitespaces

### Model Architecture
- Embedding Layer
- Two LSTM Layers (150 units each)
- Dense Layers for prediction
- Softmax activation for word prediction

### Model Training
- Uses sparse categorical cross-entropy loss
- Adam optimizer
- 50 training epochs
- Validation split for monitoring performance

## Model Performance
- Initial accuracy starts low (~5-10%)
- Gradually improves over training epochs
- Validation accuracy stabilizes around 10%

## Text Generation Method
- Uses seed text as starting point
- Generates next words probabilistically
- Implements temperature scaling for creativity
- Supports custom seed text and number of words to generate

## Usage Example
```python
seed_text = "muj se pehli se mohabbat"
generated_poetry = generate_poetry(seed_text)
print(generated_poetry)
```

## Potential Improvements
- Increase training epochs
- Experiment with model architecture
- Use larger/diverse poetry dataset
- Implement more advanced text preprocessing
- Fine-tune hyperparameters

## Limitations
- Generated text may not always make perfect semantic sense
- Requires substantial computational resources
- Performance depends heavily on training data
