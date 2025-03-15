import logging
import datetime
import random
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger("setiment_analysis_agent")
logger.setLevel(logging.INFO)

app = FastAPI

# Load FinBert model for sentiment analysis of financial texts
FINBERT_MODEL = "ipuneetr/finbert-uncased"
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

label_map = {0: "negative", 1: "neutral", 2: "positive"}

class SentimentRequest(BaseModel):
    ticker: str
    source: List[str] = ["news", "twitter", "reddit", "google", "chatGPT"]

def simulate_monte_carlo(mean_return, std_dev, days=30, simulations=1000):
    """ 
    Simulate daily returns to get a distribution of potential outcomes """

    final_prices = []
    initial_price = 100
    for _ in range(simulations):
        daily_returns = np.random.normal(mean_return, std_dev, days)
        price_path = initial_price
        for r in daily_returns:
            price_path *= (1 + r)
        final_prices.append(price_path)
    return final_prices

def calculate_var(price_distribution, confidence=0.95):
    """
    Value at Risk at the specified confidence level.
    E.g., 95% confidence means only 5% of outcomes are worse than this loss.
    """
    # Sort final prices
    sorted_prices = sorted(price_distribution)
    index = int((1 - confidence) * len(sorted_prices))
    var_price = sorted_prices[index]
    # For example, if initial price was 100, how much is lost?
    loss = 100 - var_price
    return loss

def classify_stock(sentiment_score, var_loss):
    """
    More advanced logic combining sentiment with VaR.
    """
    if sentiment_score > 0.6 and var_loss < 5:  # example thresholds
        return "BUY"
    elif sentiment_score < 0.4 and var_loss > 10:
        return "SELL"
    else:
        return "HOLD"

@app.post("/analyze-sentiment/")
def analyze_sentiment(request: SentimentRequest):
    """
    1. Fetch relevant articles/posts from each source
    2. Run each text snippet through FinBERT for sentiment classification.
    3. Aggregate average sentiment.
    4. Run a Monte Carlo simulation for risk and compute VaR.
    5. Return classification (Buy/Sell/Hold) with confidence.
    """
    try:
        # 1) Fetching mock data for demonstration
        texts = [
            f"{request.ticker} is doing great! Big upside expected.",
            f"Some negative outlook for {request.ticker} due to supply chain issues."
        ]

        # 2) Classify sentiment for each text
        total_score = 0
        for txt in texts:
            inputs = finbert_tokenizer(txt, return_tensors="pt", max_length=512, truncation=True)
            outputs = finbert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            sentiment_idx = probs.argmax()
            # Weighted average: (probability of the predicted class) * sign (+1 for positive, -1 for negative, 0 for neutral)
            if sentiment_idx == 2:  # positive
                total_score += probs[sentiment_idx]  # e.g., +0.8
            elif sentiment_idx == 0:  # negative
                total_score -= probs[sentiment_idx]  # e.g., -0.6
            else:
                total_score += 0

        avg_sentiment_score = total_score / len(texts)
        # Convert from range [0..1] for a simplified measure
        normalized_sentiment = (avg_sentiment_score + 1) / 2

        # 3) Monte Carlo for risk
        mean_return = normalized_sentiment / 100.0
        std_dev = 0.02  # toy standard deviation
        dist = simulate_monte_carlo(mean_return, std_dev, days=30, simulations=1000)
        var_loss = calculate_var(dist, confidence=0.95)

        # 4) Classify the stock
        recommendation = classify_stock(normalized_sentiment, var_loss)
        confidence = round(abs(normalized_sentiment), 2) 

        return {
            "ticker": request.ticker,
            "average_sentiment": normalized_sentiment,
            "var_loss_95": var_loss,
            "recommendation": recommendation,
            "confidence_score": confidence
        }
    except Exception as e:
        logger.error(f"Sentiment Analysis error for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
