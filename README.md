# YouTube Sentiment Analysis (MLOps)

An end-to-end Machine Learning Operations (MLOps) project that predicts the sentiment of YouTube comments. The system consists of an automated Machine Learning pipeline, a FastAPI serving backend, and a Chrome Extension frontend to analyze sentiments directly from YouTube videos.

## Features

*   **Automated ML Pipeline**: Data ingestion, preprocessing, model training, and evaluation orchestrated using **DVC** (Data Version Control).
*   **Model Tracking & Registration**: Model experiments, parameters, and metrics are tracked using **MLflow**.
*   **Model Serving API**: A **FastAPI** application that serves the LightGBM model and provides endpoints for predicting sentiments and generating analytical charts.
*   **Chrome Extension**: A browser extension that scrapes comments from a YouTube video page, sends them to the API, and displays sentiment distributions, word clouds, and trend graphs.
*   **Dependency Management**: Fast and reliable Python dependency management using **uv**.

## Tech Stack

*   **Machine Learning**: Scikit-Learn, LightGBM, NLTK
*   **MLOps**: DVC, MLflow
*   **Backend**: FastAPI, Uvicorn, Prometheus (for monitoring)
*   **Frontend**: HTML, CSS, JavaScript (Chrome Extension)
*   **Package Management**: uv

---

## Project Structure

```text
.
├── dvc.yaml                            # DVC pipeline stages
├── fastapi-app/                        # FastAPI application for serving the model
│   └── main.py                         # API endpoints and chart generation
├── mflow-setup-with-garage/            # MLflow configuration
├── notebooks/                          # Jupyter notebooks for EDA and experimentation
├── src/                                # Source code for the ML pipeline
│   ├── data/                           # Data ingestion & preprocessing scripts
│   └── model/                          # Model building, evaluation & registration scripts
├── yt-sentiment-chrome-plugin-frontend/ # Chrome extension source code
├── pyproject.toml                      # Project metadata and dependencies
└── uv.lock                             # Lockfile for reproducible environments
```

---

## Setup Instructions

### 1. Install Dependencies

The project uses `uv` for dependency management. To install the required packages:

```bash
uv sync --frozen
```

### 2. Run the ML Pipeline

The entire machine learning pipeline (data ingestion, preprocessing, training, and evaluation) is defined in `dvc.yaml`. Run it using DVC:

```bash
dvc repro
```
*This will output `lgbm_model.pkl` and `tfidf_vectorizer.pkl` which are used by the FastAPI app.*

### 3. Start MLflow

To start the MLflow tracking server (with garage server):

```bash
docker-compose up -d
```

You can view the MLflow UI by navigating to:
```
http://localhost:5000
```

### 4. Run the FastAPI Server

Once the model is built via the DVC pipeline, you can start the FastAPI application:

```bash
cd fastapi-app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The API documentation will be available at `http://localhost:8000/docs`.

### 5. Install the Chrome Extension

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** in the top right corner.
3. Click on **Load unpacked** in the top left.
4. Select the `yt-sentiment-chrome-plugin-frontend` directory.
5. The extension will now be available in your browser toolbar. Open a YouTube video and click the extension icon to analyze the comments!

---

## API Endpoints

The FastAPI backend exposes the following key endpoints:

*   `POST /predict`: Predicts the sentiment of a list of comments.
*   `POST /predict_with_timestamps`: Predicts sentiment while retaining timestamp information.
*   `POST /generate_chart`: Generates a pie chart of sentiment distribution.
*   `POST /generate_wordcloud`: Generates a word cloud from the comments.
*   `POST /generate_trend_graph`: Generates a line graph showing sentiment trends over time.