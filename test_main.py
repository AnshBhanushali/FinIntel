import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Dummy Redis client that avoids real network calls.
class DummyRedis:
    def get(self, key):
        return None
    def setex(self, key, seconds, value):
        pass

# Async dummy functions for external dependencies in investment_guidance.
async def dummy_fetch_price_data_with_cache(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=10, freq="D")
    data = {"Close": np.linspace(100, 110, len(dates))}
    return pd.DataFrame(data, index=dates)

async def dummy_post_json(url: str, payload: dict):
    return {"average_sentiment": 0.75, "recommendation": "BUY"}

async def dummy_ollama_query(prompt: str) -> str:
    return "Market outlook: Bullish and steady."

@pytest.fixture(autouse=True)
def patch_investment_guidance(monkeypatch):
    # Patch the redis_client in investment_guidance to use DummyRedis.
    monkeypatch.setattr("investment_guidance.redis_client", DummyRedis())
    # Patch the external async functions in investment_guidance.
    monkeypatch.setattr("investment_guidance.fetch_price_data_with_cache", dummy_fetch_price_data_with_cache)
    monkeypatch.setattr("investment_guidance.post_json", dummy_post_json)
    monkeypatch.setattr("investment_guidance.ollama_query", dummy_ollama_query)

def test_root():
    response = client.get("/")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data == {"message": "Welcome to the Unified AI Agents API. Visit /docs for API documentation."}

def test_data_analysis():
    response = client.post(
        "/analyze-technical/",
        json={"ticker": "AAPL", "start_date": "2023-01-01", "end_date": "2023-03-01"}
    )
    assert response.status_code == 200, response.text

def test_fundamental_analysis():
    response = client.post(
        "/fundamentals/analyze-fundamentals/",
        json={"ticker": "AAPL", "year": 2024}
    )
    assert response.status_code == 200, response.text

def test_sentiment_analysis():
    response = client.post(
        "/sentiment/analyze-sentiment/",
        json={"ticker": "AAPL", "sources": ["news", "twitter"]}
    )
    assert response.status_code == 200, response.text

def test_investment_guidance():
    response = client.post(
        "/investment-guidance/get-investment-guidance",
        json={
            "tickers": ["AAPL", "MSFT"],
            "initial_capital": 10000,
            "risk_tolerance": "medium"
        }
    )
    assert response.status_code == 200, response.text
    data = response.json()
    # Verify that key parts of the response exist.
    assert "allocation" in data
    assert "expected_annual_return" in data
    assert "annual_volatility" in data
    assert "sharpe_ratio" in data
    assert "market_outlook" in data
