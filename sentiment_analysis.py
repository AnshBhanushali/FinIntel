import logging
import random
import numpy as np
import torch
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("sentiment_analysis_agent")
logger.setLevel(logging.INFO)

app = FastAPI()

@app.get("/")
def root():
    """API health check route."""
    return {"message": "Welcome to the Sentiment Analysis API (FinBERT-only)"}

# Load FinBERT model
FINBERT_MODEL = "ProsusAI/finbert"

try:
    finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
except Exception as e:
    logger.error(f"Error loading FinBERT model: {e}")
    raise RuntimeError("Failed to load FinBERT. Check internet connection or model availability.")

label_map = {0: "negative", 1: "neutral", 2: "positive"}

class SentimentRequest(BaseModel):
    ticker: str
    sources: List[str] = ["news", "twitter"]

def simulate_monte_carlo(mean_return: float, std_dev: float, days: int = 30, simulations: int = 1000):
    """
    Simulate daily returns to get a distribution of potential outcomes.
    Ensure that returned prices are Python floats (avoid NumPy float32).
    """
    final_prices = []
    initial_price = 100.0
    for _ in range(simulations):
        daily_returns = np.random.normal(float(mean_return), float(std_dev), days)
        price_path = initial_price
        for r in daily_returns:
            price_path *= (1.0 + float(r))
        final_prices.append(float(price_path))
    return final_prices

def calculate_var(price_distribution: list, confidence: float = 0.95):
    """Value at Risk (VaR) estimation, ensuring we return a Python float."""
    sorted_prices = sorted(price_distribution)
    index = int((1 - confidence) * len(sorted_prices))
    var_price = float(sorted_prices[index])
    loss = float(100.0 - var_price)
    return loss

def classify_stock(sentiment_score: float, var_loss: float):
    """
    Classifies a stock based on sentiment and risk.
    Return: "BUY", "SELL", or "HOLD".
    """
    if sentiment_score > 0.6 and var_loss < 5:
        return "BUY"
    elif sentiment_score < 0.5 and var_loss > 10:
        return "SELL"
    else:
        return "HOLD"

@app.post("/analyze-sentiment")
def analyze_sentiment(request: SentimentRequest):
    """
    FinBERT-only sentiment analysis:
    1. Generate mock headlines for the given ticker.
    2. Classify sentiment using FinBERT (positive/negative).
    3. Run a Monte Carlo simulation for Value at Risk.
    4. Return classification (Buy/Sell/Hold) with confidence.
    """
    try:
        # 1) Mock data for demonstration
        texts = [
            f"{request.ticker} is doing great! Big upside expected.",
            f"Some negative outlook for {request.ticker} due to supply chain issues."
        ]

        if not texts:
            raise HTTPException(status_code=400, detail="No sentiment data available.")

        # 2) FinBERT Sentiment Classification
        total_score = 0.0
        for txt in texts:
            try:
                inputs = finbert_tokenizer(txt, return_tensors="pt", max_length=512, truncation=True)
                outputs = finbert_model(**inputs)

                logits = outputs.logits.squeeze(0)
                probs = torch.softmax(logits, dim=0).detach().numpy()

                sentiment_idx = int(probs.argmax())
                sentiment_prob = float(probs[sentiment_idx])

                # If positive => add; if negative => subtract
                if sentiment_idx == 2:  # positive
                    total_score += sentiment_prob
                elif sentiment_idx == 0:  # negative
                    total_score -= sentiment_prob

            except Exception as e:
                logger.error(f"Error processing text '{txt}' with FinBERT: {e}")
                continue

        if total_score == 0.0:
            # If total_score is 0, no strong positive or negative found
            # Return neutral or raise an error as per your business logic
            logger.warning("FinBERT sentiment analysis yielded total_score of 0.")
            return {
                "ticker": request.ticker,
                "average_sentiment": 0.0,
                "var_loss_95": 0.0,
                "recommendation": "HOLD",
                "confidence_score": 0.0
            }

        avg_sentiment_score = float(total_score / len(texts))
        normalized_sentiment = float((avg_sentiment_score + 1.0) / 2.0)

        # 3) Monte Carlo for risk analysis
        mean_return = float(normalized_sentiment / 100.0)
        std_dev = 0.02
        dist = simulate_monte_carlo(mean_return, std_dev, days=30, simulations=1000)
        var_loss = float(calculate_var(dist, confidence=0.95))

        # 4) Classify the stock
        recommendation = classify_stock(normalized_sentiment, var_loss)
        confidence = round(abs(normalized_sentiment), 2)

        return {
            "ticker": request.ticker,
            "average_sentiment": normalized_sentiment,
            "var_loss_95": var_loss,
            "recommendation": recommendation,
            "confidence_score": confidence,
            "finbert_texts": texts
        }

    except Exception as e:
        logger.error(f"Sentiment Analysis error for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
