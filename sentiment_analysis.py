import logging
import datetime
import random
import numpy as np
import pandas as pd
import requests
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("sentiment_analysis_agent")
logger.setLevel(logging.INFO)

app = FastAPI()

# Load a FinBERT model
FINBERT_MODEL = "ipuneetr/finbert-uncased"
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

label_map = {0: "negative", 1: "neutral", 2: "positive"}

class SentimentRequest(BaseModel):
    ticker: str
    sources: List[str] = ["news", "twitter"]

def simulate_monte_carlo(mean_return, std_dev, days=30, simulations=1000):
    """Simulate daily returns to get a distribution of potential outcomes."""
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
    """Value at Risk (VaR) estimation."""
    sorted_prices = sorted(price_distribution)
    index = int((1 - confidence) * len(sorted_prices))
    var_price = sorted_prices[index]
    loss = 100 - var_price  
    return loss

def classify_stock(sentiment_score, var_loss):
    """Classifies a stock based on sentiment and risk."""
    if sentiment_score > 0.6 and var_loss < 5:
        return "BUY"
    elif sentiment_score < 0.4 and var_loss > 10:
        return "SELL"
    else:
        return "HOLD"

def query_ollama(prompt: str):
    """Query Ollama for advanced sentiment and summary analysis."""
    url = "http://localhost:11434/run"  
    payload = {
        "model": "llama2",  
        "prompt": prompt
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logger.error(f"Ollama query failed with status {response.status_code}")
            return f"Ollama query failed with status {response.status_code}"
        return response.json().get("output", "No response from Ollama.")
    except Exception as e:
        logger.error(f"Ollama query exception: {e}")
        return f"Ollama query encountered an exception: {e}"

@app.post("/analyze-sentiment/")
def analyze_sentiment(request: SentimentRequest):
    """
    Enhanced sentiment analysis:
    1. Fetch relevant articles/posts (mocked here).
    2. Use FinBERT for classification.
    3. Query Ollama for summarization & advanced sentiment.
    4. Aggregate scores and run Monte Carlo simulation.
    5. Return classification (Buy/Sell/Hold) with confidence.
    """
    try:
        # 1) Fetching mock data
        texts = [
            f"{request.ticker} is doing great! Big upside expected.",
            f"Some negative outlook for {request.ticker} due to supply chain issues."
        ]

        # 2) FinBERT Sentiment Classification
        total_score = 0
        for txt in texts:
            inputs = finbert_tokenizer(txt, return_tensors="pt", max_length=512, truncation=True)
            outputs = finbert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            sentiment_idx = probs.argmax()
            if sentiment_idx == 2:  
                total_score += probs[sentiment_idx]  
            elif sentiment_idx == 0:  
                total_score -= probs[sentiment_idx]  
            else:
                total_score += 0  

        avg_sentiment_score = total_score / len(texts)
        normalized_sentiment = (avg_sentiment_score + 1) / 2  

        # 3) Query Ollama for more advanced sentiment analysis
        ollama_prompt = (
            f"Analyze financial sentiment based on these headlines:\n{texts}\n"
            f"Summarize sentiment impact on {request.ticker} and rate it between 0 (negative) to 1 (positive)."
        )
        ollama_response = query_ollama(ollama_prompt)

        # Extract Ollama's rating (fallback to FinBERT if it fails)
        try:
            ollama_score = float(re.search(r"Sentiment Score: (\d\.\d+)", ollama_response).group(1))
        except:
            ollama_score = normalized_sentiment  

        # 4) Monte Carlo for risk analysis
        mean_return = ollama_score / 100.0  
        std_dev = 0.02  
        dist = simulate_monte_carlo(mean_return, std_dev, days=30, simulations=1000)
        var_loss = calculate_var(dist, confidence=0.95)

        # 5) Classify the stock
        recommendation = classify_stock(ollama_score, var_loss)
        confidence = round(abs(ollama_score), 2)  

        return {
            "ticker": request.ticker,
            "average_sentiment": ollama_score,
            "var_loss_95": var_loss,
            "recommendation": recommendation,
            "confidence_score": confidence,
            "ollama_analysis": ollama_response
        }
    except Exception as e:
        logger.error(f"Sentiment Analysis error for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
