import os
import logging
import asyncio
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
import redis  # pip install redis

from pypfopt import EfficientFrontier, risk_models, expected_returns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("investment_guidance")

app = FastAPI(title="Production Investment Guidance API")

# Environment variables and Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Endpoints for other services
DATA_AGENT_URL = os.getenv("DATA_AGENT_URL", "http://data_analysis_agent:8000/analyze-technical")
SENTIMENT_AGENT_URL = os.getenv("SENTIMENT_AGENT_URL", "http://sentiment_analysis_agent:8000/analyze-sentiment")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/run")

class GuidanceRequest(BaseModel):
    tickers: List[str]
    initial_capital: float = 10000.0
    risk_tolerance: Optional[str] = "medium"  # Options: low, medium, high

def cache_key_for_ticker(ticker: str, start_date: str, end_date: str) -> str:
    return f"historical:{ticker}:{start_date}:{end_date}"

async def fetch_price_data_with_cache(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    key = cache_key_for_ticker(ticker, start_date, end_date)
    cached = redis_client.get(key)
    if cached:
        logger.info(f"Cache hit for {ticker}")
        data_dict = json.loads(cached)
        return pd.DataFrame.from_dict(data_dict)
    logger.info(f"Fetching data for {ticker} from yfinance")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for ticker {ticker}")
    # Cache the result (serialize DataFrame to JSON-compatible format)
    redis_client.setex(key, 3600, df.to_json())  # Cache for 1 hour
    return df

@app.post("/get-investment-guidance")
async def get_investment_guidance(request: GuidanceRequest):
    """
    Production-ready investment guidance service with Redis caching.
    
    1. Retrieve historical price data (cached in Redis).
    2. Compute expected returns and optimize portfolio using PyPortfolioOpt.
    3. Fetch sentiment data and generate market outlook via an LLM.
    """
    try:
        # Retrieve historical price data for each ticker (with caching)
        combined_close = pd.DataFrame()
        for ticker in request.tickers:
            df = await fetch_price_data_with_cache(ticker, "2020-01-01", "2025-01-01")
            combined_close[ticker] = df["Close"]

        if combined_close.empty:
            raise HTTPException(status_code=400, detail="No valid price data retrieved.")

        # Compute expected returns and covariance
        mu = expected_returns.mean_historical_return(combined_close)
        S = risk_models.sample_cov(combined_close)

        # Optimize portfolio
        ef = EfficientFrontier(mu, S)
        if request.risk_tolerance.lower() == "high":
            raw_weights = ef.max_quadratic_utility(risk_aversion=0.5)
        elif request.risk_tolerance.lower() == "low":
            raw_weights = ef.min_volatility()
        else:
            raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        expected_ret, volatility, sharpe = ef.portfolio_performance()

        allocations = {ticker: round(weight * request.initial_capital, 2) for ticker, weight in cleaned_weights.items()}

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
                    "recommendation": sentiment_results[idx].get("recommendation")
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

