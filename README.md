# Hate Speech Detection using LSTM

## Project Overview
This project implements a deep learning model based on Long Short-Term Memory (LSTM) networks to classify tweets into three categories:
- Hate Speech
- Offensive Language
- Neither

The goal is to automatically detect harmful or offensive content in social media text using Natural Language Processing (NLP) techniques.

---

## Dataset
The dataset consists of **24,783 labeled tweets** with two columns:
- `tweet`: textual content
- `class`: label (0 = Hate Speech, 1 = Offensive Language, 2 = Neither)

The dataset is stored at:

```bash

data/hate_speech.csv

```

---

## Model Architecture
- Embedding Layer
- LSTM Layer
- Dropout Layer
- Dense Layer with Softmax activation

The model was trained using categorical cross-entropy loss and the Adam optimizer.  
The best model was selected based on minimum validation loss.

---

## Tools & Libraries
- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- NLTK
- Matplotlib
- Jupyter Notebook

---

## Installation

### 1. Install required libraries
```bash
pip install -r requirements.txt
```
### 2. Download required NLTK resources
Run the following once in Python:
```bash
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

```
## How to Run

### Training
1. Open `notebooks/training.ipynb`
2. Run all cells sequentially to:
   - Preprocess the dataset
   - Train the LSTM model
   - Evaluate validation performance
   - Save the trained model and tokenizer
---

## Model Saving & Loading

The trained model is saved using the native Keras format:
```bash
models/hate_speech_lstm.keras
```

The tokenizer used during training is saved separately:

```bash
models/tokenizer.pkl
```

These files allow the model to be loaded later without retraining.

---

## Results Summary

- Best validation accuracy ≈ **86%**
- Best validation loss ≈ **0.43**
- Strong performance on *Offensive Language* and *Neither* classes
- Expected confusion between *Hate Speech* and *Offensive Language* due to semantic similarity and class imbalance

---
