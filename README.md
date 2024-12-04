
# Text Analysis on Indonesian Languages (Topic 3)

## Problem Statement

This project focuses on text analysis for the NusaX dataset, which is a multilingual parallel corpus encompassing various Indonesian languages. Our goal is to tackle two major tasks:
1. **Machine Translation**: Translating text from one language to another.
2. **Sentiment Analysis**: Classifying text sentiment as positive, negative, or neutral using labeled datasets.

The specific languages used for these tasks include Indonesian (Bahasa) and Javanese, with the main objective being to evaluate and improve upon existing methods for text translation and sentiment classification within these languages.

## Objectives

1. **Machine Translation Task**:
   - Perform translation between:
     - Indonesian (Bahasa) ↔ English
     - Indonesian (Bahasa) ↔ Javanese
   - Develop a system for translating between these language pairs using tokenization, encoding, and model training.
   - Evaluate translation accuracy and quality using test cases and benchmarks.

2. **Sentiment Analysis Task**:
   - Perform sentiment analysis on texts in the Indonesian dataset using a range of models:
     - Support Vector Machines (SVM)
     - Multi-Layer Perceptron (MLP)
     - Convolutional Neural Networks (CNN)
     - Transformer-based models (like BERT)
   - Visualize and analyze classification results using confusion matrices.
   - Employ pre-trained NLP models such as VADER and Word2Vec for comparison and deeper analysis.

## Dataset Overview

The dataset used in this project is the NusaX corpus, which contains parallel data across 12 languages, including Indonesian and Javanese. The data consists of:
- Textual data in the form of sentences, translated and labeled by native speakers.
- Sentiment labels for the sentiment analysis task.

## Approach

### Data Preprocessing
- **Tokenization**: Text data is tokenized using appropriate tokenizers, such as the BERT tokenizer for complex tasks, or simpler methods (like splitting text) for demonstration.
- **Padding and Truncation**: To ensure uniform input sizes, sequences are padded or truncated to a consistent length.

### Machine Translation
- Implemented models for translation between specified language pairs.
- Evaluated translation quality using metrics such as BLEU scores.

### Sentiment Analysis
- Developed and trained models on the sentiment-labeled data using various approaches (SVM, MLP, CNN, Transformers).
- Visualized results using confusion matrices and analyzed misclassifications.

### Key Features
- Utilizes a parallel dataset for multilingual text analysis.
- Comprehensive model evaluation and comparison across different methods.
- Custom preprocessing steps for effective tokenization and data preparation.

## Usage

1. **Requirements**:
   - All requirements are in the nlp_7008.yml file
   ```bash
   conda env create -f nlp_7008.yml
   conda activate nlp_7008
   ```

2. **Running the Project**:
   - Prepare the dataset as per the requirements (ensure the CSV files are correctly formatted).
   - Train models for machine translation and sentiment analysis by following the provided scripts.

3. **Example Commands**:
- We have integrated all models used for MT tasks in this project into train_MT.py, here  are some examples:
   ```bash
   # Training a transformer to translation
   python train_MT.py --model transformer --source_language indonesian --target_language english
   # Training a LSTM to translation
   python train_MT.py --model LSTM --source_language indonesian --target_language english 
   # Training a CNN to translation
   python train_MT.py --model CNN --source_language indonesian --target_language english
   # Training a RNN to translation
   python train_MT.py --model RNN --source_language indonesian --target_language english 
   ```
 - We have tried to develop the LSTM and RNN model. They showed that without the Word2Vec, it performed bad. So here we offered the choices of --model LSTM and --model RNN, but not recommend to use them.
   ```bash
   # Running sentiment analysis models
   python train_senti.py --model MLP
   python train_senti.py --model CNN
   python train_senti.py --model SVM
   ```
 - If you would like to use the Word2Vec to compare the performance with MLP, CNN and SVM, you should put the file [Fasttext](https://fasttext.cc/docs/en/crawl-vectors.html) (Indonesian, in the format of .vec) into the directionary word_2_vec
  ```bash
      cd word_2_vec
      # Three different model
      python word_2_vec_mlp.py
      python word_2_vec_cnn.py
      python word_2_vec_svm.py
   ```
  
  ```bash
   # Training a Transformer model for sentiment analysis
   python transformer.py

   # Running pre-trained IndoBERT or DistilBert
   python senti_pretrained_distilbert.py  --model indobert
   python senti_pretrained_distilbert.py  --model distilbert
   ```

## Results and Evaluation

- Detailed performance metrics, including accuracy, precision, recall, and F1-score for sentiment analysis tasks.
- Translation performance is measured using BLEU scores and qualitative assessment.
