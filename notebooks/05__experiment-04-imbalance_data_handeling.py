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
    # Set or create an experiment
    mlflow.set_experiment("Exp 4 - Handling Imbalanced Data")
    return


@app.cell
def _():
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import mlflow.sklearn
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    return (
        ADASYN,
        RandomForestClassifier,
        RandomUnderSampler,
        SMOTE,
        SMOTEENN,
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
    ADASYN,
    RandomForestClassifier,
    RandomUnderSampler,
    SMOTE,
    SMOTEENN,
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
    def run_imbalanced_experiment(imbalance_method):
        ngram_range = (1, 3)  # Trigram setting
        max_features = 10000  # Set max_features to 1000 for TF-IDF
        X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)  # Step 4: Train-test split before vectorization and resampling
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        if imbalance_method == 'class_weights':  # Step 2: Vectorization using TF-IDF, fit on training data only
            class_weight = 'balanced'
        else:  # Fit on training data
            class_weight = None  # Transform test data
            if imbalance_method == 'oversampling':
                smote = SMOTE(random_state=42)  # Step 3: Handle class imbalance based on the selected method (only applied to the training set)
                X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)
            elif imbalance_method == 'adasyn':  # Use class_weight in Random Forest
                adasyn = ADASYN(random_state=42)
                X_train_vec, y_train = adasyn.fit_resample(X_train_vec, y_train)
            elif imbalance_method == 'undersampling':  # Do not apply class_weight if using resampling
                rus = RandomUnderSampler(random_state=42)
                X_train_vec, y_train = rus.fit_resample(X_train_vec, y_train)  # Resampling Techniques (only apply to the training set)
            elif imbalance_method == 'smote_enn':
                smote_enn = SMOTEENN(random_state=42)
                X_train_vec, y_train = smote_enn.fit_resample(X_train_vec, y_train)
        with mlflow_1.start_run() as run:
            mlflow_1.set_tag('mlflow.runName', f'Imbalance_{imbalance_method}_RandomForest_TFIDF_Trigrams')
            mlflow_1.set_tag('experiment_type', 'imbalance_handling')
            mlflow_1.set_tag('model_type', 'RandomForestClassifier')
            mlflow_1.set_tag('description', f'RandomForest with TF-IDF Trigrams, imbalance handling method={imbalance_method}')
            mlflow_1.log_param('vectorizer_type', 'TF-IDF')
            mlflow_1.log_param('ngram_range', ngram_range)
            mlflow_1.log_param('vectorizer_max_features', max_features)
            n_estimators = 200
            max_depth = 15
            mlflow_1.log_param('n_estimators', n_estimators)  # Step 5: Define and train a Random Forest model
            mlflow_1.log_param('max_depth', max_depth)
            mlflow_1.log_param('imbalance_method', imbalance_method)  # Set tags for the experiment and run
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight=class_weight)
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow_1.log_metric('accuracy', accuracy)  # Add a description
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics in classification_rep.items():
                if isinstance(metrics, dict):  # Log vectorizer parameters
                    for metric, value in metrics.items():
                        mlflow_1.log_metric(f'{label}_{metric}', value)
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')  # Log Random Forest parameters
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix: TF-IDF Trigrams, Imbalance={imbalance_method}')
            confusion_matrix_filename = f'confusion_matrix_{imbalance_method}.png'
            plt.savefig(confusion_matrix_filename)
            mlflow_1.log_artifact(confusion_matrix_filename)
            plt.close()
            mlflow_1.sklearn.log_model(model, f'random_forest_model_tfidf_trigrams_imbalance_{imbalance_method}')  # Initialize and train the model
    imbalance_methods = ['class_weights', 'oversampling', 'adasyn', 'undersampling', 'smote_enn']
    for method in imbalance_methods:
    # Step 7: Run experiments for different imbalance methods
        run_imbalanced_experiment(method)  # Step 6: Make predictions and log metrics  # Log accuracy  # Log classification report  # Log confusion matrix  # Log the model
    return


if __name__ == "__main__":
    app.run()
