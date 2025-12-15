# Hate Speech Detection using LSTM

## Project Overview
This project implements a deep learning model to classify tweets into:
- Hate Speech
- Offensive Language
- Neither

## Dataset
The dataset consists of 19,826 labeled tweets and is stored in `data/hate_speech.csv`.

## Model
- Embedding Layer
- LSTM
- Dense Softmax Output

## Tools & Libraries
- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Jupyter Notebook

## How to Run
1. Open the notebook in `notebook/`
2. Run all cells sequentially

## Author
Hesham El Afandi

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