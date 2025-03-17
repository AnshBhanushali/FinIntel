import os
import logging
import asyncio
import json
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Import Hugging Face Summarization
from transformers import pipeline

# Load environment variables (optional)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("investment_guidance")

app = FastAPI(title="Production Investment Guidance API")

class GuidanceRequest(BaseModel):
    tickers: List[str]
    initial_capital: float = 10000.0
    risk_tolerance: Optional[str] = "medium"  # low, medium, high

def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical stock price data with yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for {ticker} in the given date range"
            )
        # Compute an average price
        df["price"] = df[["Open", "High", "Low", "Close"]].mean(axis=1)
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for {ticker}")

def generate_stock_graph(df: pd.DataFrame, ticker: str) -> str:
    """Generates a simple stock price graph as a base64-encoded PNG."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label=f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.title(f"{ticker} Stock Price Trend")
    plt.legend()
    plt.grid()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    encoded_img = base64.b64encode(img_buffer.read()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{encoded_img}"

# 1) Initialize a Hugging Face pipeline for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_market_outlook(summary_prompt: str) -> str:
    """
    Generates a short "Market Outlook" by summarizing prompt text
    using a Hugging Face summarization model.
    """
    try:
        # The summarizer expects a string to summarize
        summary_output = summarizer(
            summary_prompt, 
            max_length=60, 
            min_length=20, 
            do_sample=False
        )
        # Return the summary text
        return summary_output[0]["summary_text"]
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return "Unable to generate market outlook at this time."

@app.post("/investment-guidance/get-investment-guidance")
async def get_investment_guidance(request: GuidanceRequest):
    """
    1. Retrieve daily price data
    2. Compute expected returns and optimize the portfolio
    3. (Optional) Fetch sentiment data concurrently
    4. Generate a market outlook using a BERT/BART summarizer
    5. Return graphs and details
    """
    try:
        combined_prices = pd.DataFrame()
        stock_graphs = {}

        # Fetch price data
        for ticker in request.tickers:
            df = fetch_price_data(ticker, "2023-01-01", "2024-01-01")
            if "price" not in df.columns:
                raise HTTPException(status_code=500, detail="Missing 'price' column.")
            combined_prices[ticker] = df["price"]
            stock_graphs[ticker] = generate_stock_graph(df, ticker)

        if combined_prices.empty:
            raise HTTPException(status_code=400, detail="No valid price data retrieved.")

        # Compute expected returns & covariance
        mu = expected_returns.mean_historical_return(combined_prices)
        S = risk_models.sample_cov(combined_prices)

        # Optimize portfolio
        ef = EfficientFrontier(mu, S)
        if request.risk_tolerance.lower() == "high":
            ef.max_quadratic_utility(risk_aversion=0.5)
        elif request.risk_tolerance.lower() == "low":
            ef.min_volatility()
        else:
            ef.max_sharpe()

        cleaned_weights = ef.clean_weights()
        expected_ret, volatility, sharpe = ef.portfolio_performance()

        allocations = {
            ticker: round(weight * request.initial_capital, 2)
            for ticker, weight in cleaned_weights.items()
        }

        # (Optional) Sentiment step if you have a separate sentiment endpoint...
        sentiment_overview = {}
        # For example, fetch from another route or skip

        # 4) Summarize Market Outlook via Hugging Face summarizer
        outlook_prompt = (
            f"Portfolio expected annual return: {expected_ret:.2%}, "
            f"volatility: {volatility:.2%}, Sharpe ratio: {sharpe:.2f}. "
            f"Sentiment data: {json.dumps(sentiment_overview)}. "
            f"Please provide a concise market outlook for a {request.risk_tolerance} "
            f"risk tolerance with an initial capital of {request.initial_capital}."
        )

        market_outlook_summary = generate_market_outlook(outlook_prompt)

        return {
            "tickers": request.tickers,
            "risk_tolerance": request.risk_tolerance,
            "total_capital": request.initial_capital,
            "allocation": allocations,
            "expected_annual_return": round(expected_ret * 100, 4),
            "annual_volatility": round(volatility * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sentiment": sentiment_overview,  # or skip if no sentiment
            "market_outlook": market_outlook_summary,
            "stock_graphs": stock_graphs
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Investment Guidance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
