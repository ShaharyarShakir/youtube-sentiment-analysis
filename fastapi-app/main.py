import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import pickle
from typing import List, Optional
import os
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="YouTube Sentiment Analysis API", version="1.0.0")

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CommentWithTimestamp(BaseModel):
    text: str
    timestamp: str

class PredictWithTimestampsRequest(BaseModel):
    comments: List[CommentWithTimestamp]

class PredictRequest(BaseModel):
    comments: List[str]

class SentimentCounts(BaseModel):
    sentiment_counts: dict

class WordCloudRequest(BaseModel):
    comments: List[str]

class SentimentDataPoint(BaseModel):
    timestamp: str
    sentiment: str

class TrendGraphRequest(BaseModel):
    sentiment_data: List[SentimentDataPoint]


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([w for w in comment.split() if w not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        print(f"Error preprocessing comment: {e}")
        return comment


def make_prediction(model, vectorizer, texts: list) -> list:
    preprocessed  = [preprocess_comment(c) for c in texts]
    transformed   = vectorizer.transform(preprocessed)
    dense         = transformed.toarray()
    feature_names = vectorizer.get_feature_names_out()
    df_input      = pd.DataFrame(dense, columns=feature_names)
    return model.predict(df_input).tolist()


def _load_model_from_disk(model_path: str, vectorizer_path: str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ── Global model state ────────────────────────────────────────────────────────

model = None
vectorizer = None


@app.on_event("startup")
async def startup_event():
    global model, vectorizer
    try:
        model, vectorizer = _load_model_from_disk(
            "/app/lgbm_model.pkl",
            "/app/tfidf_vectorizer.pkl",
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model failed to load: {e}")


def _check_model():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Welcome to the YouTube Sentiment Analysis API"}


@app.post("/predict")
def predict(body: PredictRequest):
    _check_model()
    if not body.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        predictions = make_prediction(model, vectorizer, body.comments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(body.comments, predictions)
    ]


@app.post("/predict_with_timestamps")
def predict_with_timestamps(body: PredictWithTimestampsRequest):
    _check_model()
    if not body.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        comments    = [item.text      for item in body.comments]
        timestamps  = [item.timestamp for item in body.comments]
        predictions = [str(p) for p in make_prediction(model, vectorizer, comments)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]


@app.post("/generate_chart")
def generate_chart(body: SentimentCounts):
    if not body.sentiment_counts:
        raise HTTPException(status_code=400, detail="No sentiment counts provided")
    try:
        labels = ['Positive', 'Neutral', 'Negative']
        sizes  = [
            int(body.sentiment_counts.get('1',  0)),
            int(body.sentiment_counts.get('0',  0)),
            int(body.sentiment_counts.get('-1', 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'w'})
        plt.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', transparent=True)
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {e}")


@app.post("/generate_wordcloud")
def generate_wordcloud(body: WordCloudRequest):
    if not body.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        text = ' '.join([preprocess_comment(c) for c in body.comments])
        wc = WordCloud(
            width=800, height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False,
        ).generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {e}")


@app.post("/generate_trend_graph")
def generate_trend_graph(body: TrendGraphRequest):
    if not body.sentiment_data:
        raise HTTPException(status_code=400, detail="No sentiment data provided")
    try:
        df = pd.DataFrame([item.model_dump() for item in body.sentiment_data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_pct    = (monthly_counts.T / monthly_counts.sum(axis=1)).T * 100
        for v in [-1, 0, 1]:
            if v not in monthly_pct.columns:
                monthly_pct[v] = 0
        monthly_pct = monthly_pct[[-1, 0, 1]]
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        plt.figure(figsize=(12, 6))
        for v in [-1, 0, 1]:
            plt.plot(monthly_pct.index, monthly_pct[v],
                     marker='o', linestyle='-',
                     label=sentiment_labels[v], color=colors[v])
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {e}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)