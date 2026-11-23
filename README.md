# AI Text Detector (Java + Weka)

Classifies English essays as human-written or AI-generated using TF-IDF features and a Naive Bayes classifier, built in Java with Weka.

## Tech Stack
Java 17, Maven, Weka 3.8.6, Apache Commons CSV

## Pipeline
1. `Preparedata` cleans the raw CSV (removes empty, duplicate and very short texts), balances classes, and writes `data/data.csv`.
2. `Train` builds TF-IDF features (top 5,000 words), trains Naive Bayes, evaluates on a 20% test split, and saves the model to `models/`.
3. `Predict` loads the model and returns human/AI probabilities for a given text.

## Dataset
6,000 balanced essays (3,000 human, 3,000 AI) sampled from a Kaggle dataset (DAIGT-style). Download it yourself and put it at `data/raw.csv`. Check the dataset's license before reuse.

## Results (held-out test split, 1,200 essays)
| Metric | Value |
|---|---|
| Accuracy | 97.75% |
| Precision (AI) | 0.983 |
| Recall (AI) | 0.971 |
| F1 (human / AI) | 0.978 / 0.977 |

Confusion matrix:

| | Predicted human |
