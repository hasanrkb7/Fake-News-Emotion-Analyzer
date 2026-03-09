# Fake News Emotion Analyzer

This is my coursework from the Natural Language Processing course at the University of Jyväskylä, from February to March 2026. It has 3 short assignments and a final project. All of them are Jupyter notebooks with PyTorch and Hugging Face Transformers.

## What is inside

| Folder | What it does |
|---|---|
| `Assignment 1` | Tokenization, stopword removal, unigram and bigram probabilities and a word frequency chart (NLTK) |
| `Assignment 2` | Fine-tunes DistilBERT for sentiment classification on the IMDb review dataset, with accuracy and F1 |
| `Assignment 3` | Gets contextual embeddings from a transformer, compares emotion words with cosine similarity and plots them with PCA |
| `Project` | 2 classifiers and a combined analysis, explained below |
| `Screenshots` | Result charts and tables from the course presentation |

## Project

The project checks if emotional language goes together with misinformation. It has 3 parts.

- Misinformation classifier: it uses the LIAR dataset of short political claims. I mapped the 6 labels to 2. True-ish is half-true, mostly-true and true. False-ish is barely-true, false and pants-fire. First I tried DistilBERT. Then I used DistilRoBERTa with class-weighted loss and early stopping. The final test result is accuracy 0.63 and macro F1 0.63.
- Emotion classifier: multi-label DistilBERT on GoEmotions (28 emotion labels, sigmoid outputs). The test macro F1 is 0.45. The accuracy is 0.97, but it is per label and it is high mostly because most labels are not there for most texts.
- Combined analysis: the emotion model labels the LIAR training claims. Then I compared the emotion profile of the true-ish and false-ish claims (`Project/results/liar_emotion_vs_truthfulness_train_summary.json`).

All the metrics are saved as JSON in `Project/results`.

## How to run

Step 1: Install the packages:

```bash
pip install torch transformers datasets evaluate accelerate scikit-learn nltk matplotlib
```

Step 2: IMDb and GoEmotions download by themselves through `datasets`, but LIAR does not. Download LIAR (Wang, 2017) into `Project/data/liar/`.

Step 3: Change the `data_files` paths in the misinformation notebook. They still point to my laptop folder `E:\JYU\...`.

I did not add the trained models and checkpoints as they are too big.
