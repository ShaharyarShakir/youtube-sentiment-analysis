# /// script
# dependencies = ["awscli", "boto3", "imbalanced-learn", "lightgbm", "mlflow", "optuna"]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell
def _():
    # packages added via marimo's package management: mlflow boto3 awscli optuna imbalanced-learn lightgbm !pip install mlflow boto3 awscli optuna imbalanced-learn lightgbm
    return


@app.cell
def _(subprocess):
    #! aws configure
    subprocess.call(['aws', 'configure'])
    return


@app.cell
def _():
    import mlflow
    # Step 2: Set up the MLflow tracking server
    mlflow.set_tracking_uri("http://ec2-54-196-109-131.compute-1.amazonaws.com:5000/")
    return (mlflow,)


@app.cell
def _(mlflow):
    # Set or create an experiment
    mlflow.set_experiment("LightGBM HP Tuning")
    return


@app.cell
def _():
    import pandas as pd

    df = pd.read_csv('/content/reddit_preprocessing.csv').dropna()
    df.shape
    return (df,)


@app.cell
def _():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from imblearn.over_sampling import SMOTE
    import mlflow.sklearn
    import optuna
    from lightgbm import LGBMClassifier
    import matplotlib.pyplot as plt

    return (
        LGBMClassifier,
        SMOTE,
        TfidfVectorizer,
        accuracy_score,
        classification_report,
        mlflow,
        optuna,
        train_test_split,
    )


@app.cell
def _(df):
    # Step 1: Remap the class labels from [-1, 0, 1] to [2, 0, 1]
    df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
    # Step 2: Remove rows where the target labels (category) are NaN
    df_1 = df.dropna(subset=['category'])
    return (df_1,)


@app.cell
def _(SMOTE, TfidfVectorizer, df_1):
    # Step 3: TF-IDF vectorizer setup
    ngram_range = (1, 3)  # Trigram
    max_features = 1000  # Set max_features to 1000
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(df_1['clean_comment'])
    y = df_1['category']
    smote = SMOTE(random_state=42)
    # Step 4: Apply SMOTE to handle class imbalance
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


@app.cell
def _(X_resampled, train_test_split, y_resampled):
    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    return X_test, X_train, y_test, y_train


@app.cell
def _(accuracy_score, classification_report, mlflow_1):
    # Function to log results in MLflow
    def log_mlflow(model_name, model, X_train, X_test, y_train, y_test, params, trial_number):
        with mlflow_1.start_run():
            mlflow_1.set_tag('mlflow.runName', f'Trial_{trial_number}_{model_name}_SMOTE_TFIDF_Trigrams')  # Log model type and trial number
            mlflow_1.set_tag('experiment_type', 'algorithm_comparison')
            mlflow_1.log_param('algo_name', model_name)
            for key, value in params.items():
                mlflow_1.log_param(key, value)  # Log algorithm name as a parameter
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)  # Log hyperparameters
            mlflow_1.log_metric('accuracy', accuracy)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics in classification_rep.items():
                if isinstance(metrics, dict):  # Train model
                    for metric, value in metrics.items():
                        mlflow_1.log_metric(f'{label}_{metric}', value)
            mlflow_1.sklearn.log_model(model, f'{model_name}_model')
            return accuracy  # Log accuracy  # Log classification report  # Log the model

    return (log_mlflow,)


@app.cell
def _(LGBMClassifier, X_test, X_train, log_mlflow, y_test, y_train):
    # Step 6: Optuna objective function for LightGBM
    def objective_lightgbm(trial):
        # Hyperparameter space to explore
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        num_leaves = trial.suggest_int('num_leaves', 20, 150)
        min_child_samples = trial.suggest_int('min_child_samples', 10, 100)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True)  # L1 regularization
        reg_lambda = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)  # L2 regularization

        # Log trial parameters
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda
        }

        # Create LightGBM model
        model = LGBMClassifier(n_estimators=n_estimators,
                               learning_rate=learning_rate,
                               max_depth=max_depth,
                               num_leaves=num_leaves,
                               min_child_samples=min_child_samples,
                               colsample_bytree=colsample_bytree,
                               subsample=subsample,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda,
                               random_state=42)

        # Log each trial as a separate run in MLflow
        accuracy = log_mlflow("LightGBM", model, X_train, X_test, y_train, y_test, params, trial.number)

        return accuracy

    return (objective_lightgbm,)


@app.cell
def _(
    LGBMClassifier,
    X_test,
    X_train,
    log_mlflow,
    objective_lightgbm,
    optuna,
    y_test,
    y_train,
):
    # Step 7: Run Optuna for LightGBM, log the best model, and plot the importance of each parameter
    def run_optuna_experiment():
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_lightgbm, n_trials=100)  # Increased to 100 trials

        # Get the best parameters
        best_params = study.best_params
        best_model = LGBMClassifier(n_estimators=best_params['n_estimators'],
                                    learning_rate=best_params['learning_rate'],
                                    max_depth=best_params['max_depth'],
                                    num_leaves=best_params['num_leaves'],
                                    min_child_samples=best_params['min_child_samples'],
                                    colsample_bytree=best_params['colsample_bytree'],
                                    subsample=best_params['subsample'],
                                    reg_alpha=best_params['reg_alpha'],
                                    reg_lambda=best_params['reg_lambda'],
                                    random_state=42)

        # Log the best model with MLflow and print the classification report
        log_mlflow("LightGBM", best_model, X_train, X_test, y_train, y_test, best_params, "Best")

        # Plot parameter importance
        optuna.visualization.plot_param_importances(study).show()

        # Plot optimization history
        optuna.visualization.plot_optimization_history(study).show()

    return (run_optuna_experiment,)


@app.cell
def _(run_optuna_experiment):
    # Run the experiment for LightGBM
    run_optuna_experiment()
    return


@app.cell
def _(best_model):
    best_model
    return


if __name__ == "__main__":
    app.run()
