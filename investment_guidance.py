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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("investment_guidance")

app = FastAPI(title="Production Investment Guidance API")

# Endpoints for other services
DATA_AGENT_URL = os.getenv("DATA_AGENT_URL", "http://data_analysis_agent:8000/analyze-technical")
SENTIMENT_AGENT_URL = os.getenv("SENTIMENT_AGENT_URL", "http://localhost:8000/sentiment/analyze-sentiment")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/run")

print("OLLAMA_URL is:", os.getenv("OLLAMA_URL", "http://localhost:11434/generate"))

class GuidanceRequest(BaseModel):
    tickers: List[str]
    initial_capital: float = 10000.0
    risk_tolerance: Optional[str] = "medium"  # Options: low, medium, high

def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock price data using Yahoo Finance (yfinance).
    """
    try:
        logger.info(f"Fetching data for {ticker} from Yahoo Finance")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker} in the given date range")

        # Compute an average price from Open, High, Low, and Close
        df["price"] = df[["Open", "High", "Low", "Close"]].mean(axis=1)
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for {ticker}")

def generate_stock_graph(df: pd.DataFrame, ticker: str) -> str:
    """
    Generates a stock price graph and returns it as a base64 string.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label=f"{ticker} Stock Price", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.title(f"{ticker} Stock Price Trend")
    plt.legend()
    plt.grid()

    # Convert plot to base64 image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    encoded_img = base64.b64encode(img_buffer.read()).decode("utf-8")
    plt.close()

    return f"data:image/png;base64,{encoded_img}"

@app.post("/get-investment-guidance")
async def get_investment_guidance(request: GuidanceRequest):
    """
    Investment guidance service using Yahoo Finance:
    1. Retrieves daily price data.
    2. Computes expected returns and optimizes the portfolio.
    3. Fetches sentiment data concurrently.
    4. Generates a market outlook using an LLM.
    5. Returns stock price graphs.
    """
    try:
        combined_prices = pd.DataFrame()
        stock_graphs = {}

        for ticker in request.tickers:
            df = fetch_price_data(ticker, "2023-01-01", "2024-01-01")
            if "price" not in df.columns:
                raise HTTPException(status_code=500, detail="Expected computed 'price' column not found in data")

            combined_prices[ticker] = df["price"]
            stock_graphs[ticker] = generate_stock_graph(df, ticker)

        if combined_prices.empty:
            raise HTTPException(status_code=400, detail="No valid price data retrieved.")

        # Compute expected returns and covariance
        mu = expected_returns.mean_historical_return(combined_prices)
        S = risk_models.sample_cov(combined_prices)

        # Optimize portfolio based on risk tolerance
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

        # Fetch sentiment data concurrently
        sentiment_overview = {}
        sentiment_tasks = []
        for ticker in request.tickers:
            s_payload = {"ticker": ticker, "sources": ["news", "twitter"]}
            sentiment_tasks.append(asyncio.create_task(post_json(SENTIMENT_AGENT_URL, s_payload)))
        sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
        for idx, ticker in enumerate(request.tickers):
            if isinstance(sentiment_results[idx], Exception):
                logger.error(f"Sentiment error for {ticker}: {sentiment_results[idx]}")
                sentiment_overview[ticker] = {"error": str(sentiment_results[idx])}
            else:
                sentiment_overview[ticker] = {
                    "average_sentiment": sentiment_results[idx].get("average_sentiment"),
                    "recommendation": sentiment_results[idx].get("recommendation"),
                }

        # Generate market outlook via LLM (Ollama)
        ollama_prompt = (
            f"Given a portfolio with an expected annual return of {expected_ret:.2f}, "
            f"annual volatility of {volatility:.2f}, and a Sharpe ratio of {sharpe:.2f}, "
            f"and considering the following sentiment overview: {sentiment_overview}, "
            f"provide a market outlook and recommended investment strategy for an initial capital of {request.initial_capital} "
            f"with a {request.risk_tolerance} risk tolerance."
        )
        llm_summary = await ollama_query(ollama_prompt)

        return {
            "risk_tolerance": request.risk_tolerance,
            "total_capital": request.initial_capital,
            "allocation": allocations,
            "expected_annual_return": round(expected_ret, 4),
            "annual_volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sentiment": sentiment_overview,
            "market_outlook": llm_summary,
            "stock_graphs": stock_graphs
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Investment Guidance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def post_json(url: str, payload: dict):
    """
    Async helper to POST JSON data to another service.
    """
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=text)
            return await resp.json()

async def ollama_query(prompt: str) -> str:
    """
    Async helper to query an LLM (e.g., Ollama) for market insights.
    """
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json={"model": "llama2", "prompt": prompt}) as resp:
            return await resp.text()
