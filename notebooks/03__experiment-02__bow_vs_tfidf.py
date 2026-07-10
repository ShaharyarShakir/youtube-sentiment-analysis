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
    return (mlflow,)


@app.cell
def _(mlflow):
    mlflow.set_experiment("Exp 2 - BoW vs TfIdf")
    return


@app.cell
def _():
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import mlflow.sklearn
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    return (
        CountVectorizer,
        RandomForestClassifier,
        TfidfVectorizer,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mlflow,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv('reddit_preprocessing.csv').dropna(subset=['clean_comment'])
    df.shape
    return (df,)


@app.cell
def _(
    CountVectorizer,
    RandomForestClassifier,
    TfidfVectorizer,
    accuracy_score,
    classification_report,
    confusion_matrix,
    df,
    mlflow_1,
    plt,
    sns,
    train_test_split,
):
    # Step 1: Function to run the experiment
    def run_experiment(vectorizer_type, ngram_range, vectorizer_max_features, vectorizer_name):
        if vectorizer_type == 'BoW':  # Step 2: Vectorization
            vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=vectorizer_max_features)
        else:
            vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=vectorizer_max_features)
        X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        with mlflow_1.start_run() as run:
            mlflow_1.set_tag('mlflow.runName', f'{vectorizer_name}_{ngram_range}_RandomForest')
            mlflow_1.set_tag('experiment_type', 'feature_engineering')
            mlflow_1.set_tag('model_type', 'RandomForestClassifier')
            mlflow_1.set_tag('description', f'RandomForest with {vectorizer_name}, ngram_range={ngram_range}, max_features={vectorizer_max_features}')  # Step 4: Define and train a Random Forest model
            mlflow_1.log_param('vectorizer_type', vectorizer_type)
            mlflow_1.log_param('ngram_range', ngram_range)  # Set tags for the experiment and run
            mlflow_1.log_param('vectorizer_max_features', vectorizer_max_features)
            n_estimators = 200
            max_depth = 15
            mlflow_1.log_param('n_estimators', n_estimators)
            mlflow_1.log_param('max_depth', max_depth)  # Add a description
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)  # Log vectorizer parameters
            accuracy = accuracy_score(y_test, y_pred)
            mlflow_1.log_metric('accuracy', accuracy)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics in classification_rep.items():
                if isinstance(metrics, dict):  # Log Random Forest parameters
                    for metric, value in metrics.items():
                        mlflow_1.log_metric(f'{label}_{metric}', value)
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')  # Initialize and train the model
            plt.title(f'Confusion Matrix: {vectorizer_name}, {ngram_range}')
            plt.savefig('confusion_matrix.png')
            mlflow_1.log_artifact('confusion_matrix.png')
            plt.close()  # Step 5: Make predictions and log metrics
            mlflow_1.sklearn.log_model(model, f'random_forest_model_{vectorizer_name}_{ngram_range}')
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    max_features = 5000  # Log accuracy
    for ngram_range in ngram_ranges:
        run_experiment('BoW', ngram_range, max_features, vectorizer_name='BoW')
    # Step 6: Run experiments for BoW and TF-IDF with different n-grams
        run_experiment('TF-IDF', ngram_range, max_features, vectorizer_name='TF-IDF')  # Log classification report  # Log confusion matrix  # Log the model  # unigrams, bigrams, trigrams  # Example max feature size  # BoW Experiments  # TF-IDF Experiments
    return


if __name__ == "__main__":
    app.run()
