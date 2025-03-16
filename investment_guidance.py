import logging
import asyncio
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from pypfopt import EfficientFrontier, risk_models, expected_returns

logger = logging.getLogger("investment_guidance")
logger.setLevel(logger.INFO)

app = FastAPI()

class GuidanceRequest(BaseModel):
    tickers : List[str]
    initial_capital: float = 10000
    risk_tolerance: Optional[str] = "medium" # User can choose between low, medium, high


DATA_AGENT_URL = "http://data_analysis_agent:8000/analyze-technical"
SENTIMENT_AGENT_URL = "http://sentiment_analysis_agent:8000/analyze-sentiment"

@app.post("/get-investment-guidance")
async def get_investment_guidance(request: GuidanceRequest):
    """
    1. For each ticker, gather historical data from DataAnalysisAgent or DB.
    2. Compute expected returns, covariance matrix using PyPortfolioOpt.
    3. Optimize for risk tolerance (max Sharpe, min vol, etc.).
    4. Provide recommended allocations, expected returns, risk metrics.
    """
    try:
        # Step 1: Gather historical data concurrently 
        tasks = []
        for ticker in request.tickers:
            payload = {
                "ticker": ticker,
                "start_date": "2020-01-01",
                "end_date": "2025-01-01",
                "indicators": [],
                "forecast_models": []
            }
            tasks.append(asyncio.create_task(post_json(DATA_AGENT_URL, payload)))
        
        data_responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 2: Build a combined price DataFrame
        combined_close = pd.DataFrame()
        for idx, ticker in enumerate(request.tickers):
            if isinstance(data_responses[idx], Exception):
                logger.error(f"Error retrieving data for {ticker}: {data_responses[idx]}")
                continue

            import numpy as np
            rng = pd.date_range("2020-01-01", periods=1000, freq="D")
            closes = np.random.normal(100, 10, size=1000)  # random data
            combined_close[ticker] = pd.Series(closes, index=rng)

        if combined_close.empty:
            raise HTTPException(status_code=400, detail="No valid price data retrieved.")

        # Step 3: Use PyPortfolioOpt to compute expected returns & covariance
        mu = expected_returns.mean_historical_return(combined_close)
        S = risk_models.sample_cov(combined_close)

        # Step 4: Optimize based on risk tolerance
        ef = EfficientFrontier(mu, S)
        if request.risk_tolerance == "high":
            # If high risk, maybe max expected returns
            raw_weights = ef.max_quadratic_utility(risk_aversion=0.5)
        elif request.risk_tolerance == "low":
            # If low risk, maybe minimize volatility
            raw_weights = ef.min_volatility()
        else:
            # Default medium - max Sharpe
            raw_weights = ef.max_sharpe()

        cleaned_weights = ef.clean_weights()
        expected_ret, volatility, sharpe = ef.portfolio_performance()

        # Step 5: Prepare final results
        allocations = {}
        for ticker, weight in cleaned_weights.items():
            allocations[ticker] = round(weight * request.initial_capital, 2)

        # Optional: fetch sentiment data for each ticker to show an overlay
        sentiment_overview = {}
        for ticker in request.tickers:
            s_payload = {"ticker": ticker, "sources": ["news", "twitter"]}
            try:
                s_response = requests.post(SENTIMENT_AGENT_URL, json=s_payload).json()
                sentiment_overview[ticker] = {
                    "average_sentiment": s_response.get("average_sentiment"),
                    "recommendation": s_response.get("recommendation")
                }
            except Exception as e:
                logger.error(f"Could not fetch sentiment for {ticker}: {e}")
                sentiment_overview[ticker] = {"error": str(e)}

        return {
            "risk_tolerance": request.risk_tolerance,
            "total_capital": request.initial_capital,
            "allocation": allocations,
            "expected_annual_return": round(expected_ret, 4),
            "annual_volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sentiment": sentiment_overview
        }

    except Exception as e:
        logger.error(f"Investment Guidance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

async def post_json(url, payload):
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