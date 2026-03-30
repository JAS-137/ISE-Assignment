import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk
#change to your own path
nltk.data.path.append("/Users/jas/nltk_data") #had to use my own download of the stop words thr api's one did not work
from nltk.corpus import stopwords

def remove_html(text: str) -> str:
    return re.sub(r"<.*?>", " ", str(text)) #removing HTML tags from code

def remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "[" +
        u"\U0001F600-\U0001F64F" +
        u"\U0001F300-\U0001F5FF" +
        u"\U0001F680-\U0001F6FF" +
        u"\U0001F1E0-\U0001F1FF" + #removing any emojis that might be present
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(" ", str(text))

def preprocess(text: str) -> str:
    text = remove_html(text)
    text = remove_emoji(text)
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip() #keeping all the text in lower case and removing any extra spaces
    return text

stop_words = stopwords.words("english")
projects = ["pytorch", "tensorflow", "keras", "incubator-mxnet", "caffe"]
num_runs = 10

results = []

for project in projects:
    for run_id in range(num_runs):
        df = pd.read_csv(f"{project}.csv").sample(frac=1, random_state=999).reset_index(drop=True)
        df["text"] = df["Title"].fillna("") + ". " + df["Body"].fillna("")
        data = df.rename(columns={"Unnamed: 0": "id", "class": "sentiment"})[
            ["id", "Number", "sentiment", "text"]
        ].copy()
        data["text"] = data["text"].apply(preprocess) #applying the preprocessing to the dataset

        X = data["text"].astype(str)
        y = data["sentiment"].astype(int)
        trainX, testX, trainY, testY = train_test_split(
            X, y,
            test_size=0.2,
            random_state=run_id,     #randomly splitting the datset into 20% test data and 80% train data randomly per run
            stratify=y
        )
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=1000,
                stop_words=stop_words
            )),
            ("sgd", SGDClassifier(random_state=run_id))  #applying the pipeline
        ])

        settings = {
            "sgd__loss": ["hinge", "log_loss"],
            "sgd__alpha": [1e-5, 1e-4, 1e-3],
            "sgd__penalty": ["l2", "l1"] #defining the loss functions for the sgd classifer to use, will find the best log function and settings
        }
        grid = GridSearchCV(
            pipe,
            settings,
            cv=5,
            scoring="f1",
            n_jobs=-1 #for the log function use f1 score to dermine best loss function
        )

        grid.fit(trainX, trainY)

        best_model = grid.best_estimator_ #find best model

        y_pred = best_model.predict(testX)

        accuracy = accuracy_score(testY, y_pred)
        precision = precision_score(testY, y_pred, average="macro", zero_division=0)
        recall = recall_score(testY, y_pred, average="macro", zero_division=0) #calculating the metrics
        f1 = f1_score(testY, y_pred, average="macro", zero_division=0)

        results.append({
            "dataset": project,
            "run_id": run_id,
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
        })

results_df = pd.DataFrame(results)
out_file = "Metrics_SGD.csv"
results_df.to_csv(out_file, index=False)