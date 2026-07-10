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
    mlflow.set_experiment("Exp 5 - ML Algos with HP Tuning")
    return


@app.cell
def _():
    import optuna
    import mlflow.sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    return (
        SMOTE,
        TfidfVectorizer,
        XGBClassifier,
        accuracy_score,
        classification_report,
        mlflow,
        optuna,
        pd,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv('reddit_preprocessing.csv').dropna()
    df.shape
    return (df,)


@app.cell
def _(
    SMOTE,
    TfidfVectorizer,
    XGBClassifier,
    accuracy_score,
    classification_report,
    df,
    mlflow_1,
    optuna,
    train_test_split,
):
    # Step 1: Remap the class labels from [-1, 0, 1] to [2, 0, 1]
    df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
    df_1 = df.dropna(subset=['category'])
    # Step 2: Remove rows where the target labels (category) are NaN
    ngram_range = (1, 3)
    max_features = 10000
    X_train, X_test, y_train, y_test = train_test_split(df_1['clean_comment'], df_1['category'], test_size=0.2, random_state=42, stratify=df_1['category'])  # Trigram setting
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)  # Set max_features to 1000 for TF-IDF
    X_train_vec = vectorizer.fit_transform(X_train)
    # Step 4: Train-test split before vectorization and resampling
    X_test_vec = vectorizer.transform(X_test)
    smote = SMOTE(random_state=42)
    # Step 2: Vectorization using TF-IDF, fit on training data only
    X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)
      # Fit on training data
    def log_mlflow(model_name, model, X_train, X_test, y_train, y_test):  # Transform test data
        with mlflow_1.start_run():
            mlflow_1.set_tag('mlflow.runName', f'{model_name}_SMOTE_TFIDF_Trigrams')
            mlflow_1.set_tag('experiment_type', 'algorithm_comparison')
            mlflow_1.log_param('algo_name', model_name)
    # Function to log results in MLflow
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)  # Log model type
            mlflow_1.log_metric('accuracy', accuracy)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics in classification_rep.items():
                if isinstance(metrics, dict):  # Log algorithm name as a parameter
                    for metric, value in metrics.items():
                        mlflow_1.log_metric(f'{label}_{metric}', value)
            mlflow_1.sklearn.log_model(model, f'{model_name}_model')  # Train model

    def objective_xgboost(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)  # Log accuracy
        max_depth = trial.suggest_int('max_depth', 3, 10)
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        return accuracy_score(y_test, model.fit(X_train_vec, y_train).predict(X_test_vec))
      # Log classification report
    def run_optuna_experiment():
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_xgboost, n_trials=30)
        best_params = study.best_params
        best_model = XGBClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'], random_state=42)
        log_mlflow('XGBoost', best_model, X_train_vec, X_test_vec, y_train, y_test)
    # Step 6: Optuna objective function for XGBoost
    # Step 7: Run Optuna for XGBoost, log the best model only
    # Run the experiment for XGBoost
    run_optuna_experiment()  # Log the model  # Get the best parameters and log only the best model  # Log the best model with MLflow, passing the algo_name as "xgboost"
    return


if __name__ == "__main__":
    app.run()
