import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import os
    import mlflow

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:3900"
    os.environ["AWS_ACCESS_KEY_ID"]      = "mlflow-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"]  = "mlflow-secret-key"
    os.environ["AWS_DEFAULT_REGION"]     = "garage"

    mlflow.set_tracking_uri("http://localhost:5000/")

    with mlflow.start_run():
        mlflow.log_param("param1", 15)
        mlflow.log_metric("metric1", 0.89)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
    df.head()
    return (df,)


@app.cell
def _(df):
    df.dropna(inplace=True)
    return


@app.cell
def _(df):
    df.drop_duplicates(inplace=True)
    return


@app.cell
def _():
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer, nltk, re, stopwords


@app.cell
def _(nltk):
    nltk.download('stopwords')
    nltk.download('wordnet')
    return


@app.cell
def _(WordNetLemmatizer, re, stopwords):
    def preprocess_comment(comment):
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment

    return (preprocess_comment,)


@app.cell
def _(df, preprocess_comment):
    df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _():
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    return (
        CountVectorizer,
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(CountVectorizer):
    # Step 1: Vectorize the comments using Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer(max_features=10000)  # Bag of Words model with a limit of 1000 features
    return (vectorizer,)


@app.cell
def _(df, vectorizer):
    X = vectorizer.fit_transform(df['clean_comment']).toarray()
    y = df['category']  # Assuming 'sentiment' is the target variable (0 or 1 for binary classification)
    return X, y


@app.cell
def _(X):
    X
    return


@app.cell
def _(X):
    X.shape
    return


@app.cell
def _(y):
    y
    return


@app.cell
def _(y):
    y.shape
    return


@app.cell
def _(mlflow_1):
    mlflow_1.set_tracking_uri('http://localhost:5000/')
    return


@app.cell
def _(mlflow_1):
    mlflow_1.set_experiment('RF Baseline')
    return


@app.cell
def _(
    RandomForestClassifier,
    X,
    accuracy_score,
    classification_report,
    confusion_matrix,
    df,
    mlflow_1,
    plt,
    sns,
    train_test_split,
    vectorizer,
    y,
):
    # Step 1: Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    with mlflow_1.start_run() as run:
        mlflow_1.set_tag('mlflow.runName', 'RandomForest_Baseline_TrainTestSplit')
        mlflow_1.set_tag('experiment_type', 'baseline')
        mlflow_1.set_tag('model_type', 'RandomForestClassifier')
        mlflow_1.set_tag('description', 'Baseline RandomForest model for sentiment analysis using BoW (CountVectorizer) + train-test split')
        mlflow_1.log_param('vectorizer_type', 'CountVectorizer')  # ---------------------------
        mlflow_1.log_param('vectorizer_max_features', vectorizer.max_features)  # Tags / metadata
        n_estimators = 200  # ---------------------------
        max_depth = 15
        mlflow_1.log_param('n_estimators', n_estimators)
        mlflow_1.log_param('max_depth', max_depth)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow_1.log_metric('accuracy', accuracy)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)  # ---------------------------
        for label, metrics in classification_rep.items():  # Vectorizer params
            if isinstance(metrics, dict):  # ---------------------------
                for metric, value in metrics.items():
                    mlflow_1.log_metric(f'{label}_{metric}', value)
        cm_path = 'confusion_matrix.png'
        conf_matrix = confusion_matrix(y_test, y_pred)  # ---------------------------
        plt.figure(figsize=(8, 6))  # Model params
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')  # ---------------------------
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        mlflow_1.log_artifact(cm_path)  # ---------------------------
        mlflow_1.sklearn.log_model(model, 'random_forest_model')  # Train model
        data_path = 'dataset.csv'  # ---------------------------
        df.to_csv(data_path, index=False)
        mlflow_1.log_artifact(data_path)
    # Final output
    print(f'Accuracy: {accuracy}')  # ---------------------------  # Predictions  # ---------------------------  # Metrics  # ---------------------------  # Confusion matrix plot  # ---------------------------  # Log model  # ---------------------------  # Save dataset artifact  # ---------------------------
    return y_pred, y_test


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _(df):
    df.to_csv('reddit_preprocessing.csv', index=False)
    return


@app.cell
def _(pd):
    pd.read_csv('reddit_preprocessing.csv').head()
    return


if __name__ == "__main__":
    app.run()
