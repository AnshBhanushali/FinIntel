import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from investment_guidance import app  # adjust if your file is named differently

client = TestClient(app)

# Async dummy functions for external dependencies

async def dummy_fetch_price_data_with_cache(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Return a dummy DataFrame with a 'Close' column for testing."""
    dates = pd.date_range(start_date, periods=10, freq="D")
    data = {"Close": np.linspace(100, 110, len(dates))}
    return pd.DataFrame(data, index=dates)

async def dummy_post_json(url: str, payload: dict):
    """Return dummy sentiment data."""
    return {"average_sentiment": 0.75, "recommendation": "BUY"}

async def dummy_ollama_query(prompt: str) -> str:
    """Return a dummy LLM summary."""
    return "Market outlook: Bullish and steady."

@pytest.fixture(autouse=True)
def patch_external_calls(monkeypatch):
    # Patch external async calls in the investment_guidance module.
    monkeypatch.setattr("investment_guidance.fetch_price_data_with_cache", dummy_fetch_price_data_with_cache)
    monkeypatch.setattr("investment_guidance.post_json", dummy_post_json)
    monkeypatch.setattr("investment_guidance.ollama_query", dummy_ollama_query)

def test_get_investment_guidance_success():
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "initial_capital": 10000,
        "risk_tolerance": "medium"
    }
    response = client.post("/get-investment-guidance", json=payload)
    # Check that we get a 200 OK
    assert response.status_code == 200, response.text
    data = response.json()
    # Validate key parts of the response
    assert "allocation" in data
    assert "expected_annual_return" in data
    assert "annual_volatility" in data
    assert "sharpe_ratio" in data
    assert "market_outlook" in data
    assert data["market_outlook"] == "Market outlook: Bullish and steady."

def test_get_investment_guidance_no_tickers():
    payload = {
        "tickers": [],
        "initial_capital": 10000,
        "risk_tolerance": "medium"
    }
    response = client.post("/get-investment-guidance", json=payload)
    # Expect the endpoint to return a 400 status code along with a detail message.
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "No valid price data retrieved."
