# AI Text Detector (Java + Weka)

Classifies English essays as human-written or AI-generated using TF-IDF features and a Naive Bayes classifier, built in Java with Weka. Includes a small Spring Boot web interface.

## Tech Stack
Java 17, Maven, Weka 3.8.6, Apache Commons CSV, Spring Boot 3

## Pipeline
1. `Preparedata` cleans the raw CSV (removes empty, duplicate and very short texts), balances classes, and writes `data/data.csv`.
2. `Train` builds TF-IDF features (top 5,000 words), trains Naive Bayes, evaluates on a 20% test split, and saves the model to `models/`.
3. `Predict` loads the model and returns human/AI probabilities for a given text.
4. `WebApp` and `DetectController` (Spring Boot) expose the model through a REST endpoint and a simple web page.

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

| | Predicted human | Predicted AI |
|---|---|---|
| Actual human | 600 | 10 |
| Actual AI | 17 | 573 |

## How to run
1. Download the dataset to `data/raw.csv`.
2. Set `BASE` in `Train.java` and `Predict.java` to your project folder path.
3. Run `Preparedata`, then `Train`. This creates the model files in `models/`.
4. Run `Predict` for a quick console test.

## Web interface
1. Run `WebApp` (Spring Boot).
2. Open `http://localhost:8080`.
3. Paste an English text (30 or more words) and click **Check** to see the human/AI probabilities.

The endpoint is `POST /api/predict` with a JSON body like `{"text": "..."}`. It returns `{"human": 12.3, "ai": 87.7, "words": 120}`.

## Project structure
