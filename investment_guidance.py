import os
import logging
import asyncio
import json
import requests
import pandas as pd
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
# IMPORTANT: Update the LLM endpoint for Ollama version 0.5.12
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/run")


load_dotenv()
print("OLLAMA_URL is:", os.getenv("OLLAMA_URL", "http://localhost:11434/generate"))


class GuidanceRequest(BaseModel):
    tickers: List[str]
    initial_capital: float = 10000.0
    risk_tolerance: Optional[str] = "medium"  # Options: low, medium, high

def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches daily price data from Alpha Vantage using TIME_SERIES_DAILY (free plan).
    Computes an average price from open, high, low, and close.
    """
    logger.info(f"Fetching data for {ticker} from Alpha Vantage (TIME_SERIES_DAILY)")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Alpha Vantage API key not provided")
    
    # Use TIME_SERIES_DAILY with compact output for ~100 days of data
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}&apikey={api_key}&outputsize=compact"
    )
    response = requests.get(url)
    logger.info(f"Alpha Vantage raw response for {ticker}: {response.text}")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from Alpha Vantage")
    
    data = response.json()
    if "Time Series (Daily)" not in data:
        logger.error(f"Alpha Vantage response for {ticker} does not contain 'Time Series (Daily)'. Full response: {data}")
        raise HTTPException(status_code=500, detail="Alpha Vantage response format error")
    
    ts_data = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts_data, orient="index")
    
    # Convert index to datetime and sort
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Ensure required columns exist
    required_keys = ["1. open", "2. high", "3. low", "4. close"]
    for key_ in required_keys:
        if key_ not in df.columns:
            logger.error(f"DataFrame columns for {ticker}: {list(df.columns)}")
            raise HTTPException(status_code=500, detail=f"Expected '{key_}' column not found in data")
        df[key_] = df[key_].astype(float)
    
    # Compute an average price from open, high, low, and close
    df["price"] = df[required_keys].mean(axis=1)
    
    # Filter by date range
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for ticker {ticker} in the given date range")
    
    logger.info(f"Fetched DataFrame columns for {ticker}: {list(df.columns)}")
    return df

@app.post("/get-investment-guidance")
async def get_investment_guidance(request: GuidanceRequest):
    """
    Investment guidance service using Alpha Vantage (free plan):
    1. Retrieves daily price data (computes average price).
    2. Computes expected returns and optimizes the portfolio using PyPortfolioOpt.
    3. Fetches sentiment data concurrently.
    4. Generates a market outlook using an LLM.
    """
    try:
        combined_prices = pd.DataFrame()
        for ticker in request.tickers:
            df = fetch_price_data(ticker, "2020-01-01", "2025-01-01")
            if "price" not in df.columns:
                raise HTTPException(status_code=500, detail="Expected computed 'price' column not found in data")
            combined_prices[ticker] = df["price"]

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
            "market_outlook": llm_summary
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
    Async helper to query an LLM (e.g., Ollama) for narrative insights.
    """
    import aiohttp
    payload = {"model": "llama2", "prompt": prompt}
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Ollama query failed with status {resp.status}: {text}")
                return "LLM query failed."
            data = await resp.json()
            return data.get("output", "No output from LLM.")
