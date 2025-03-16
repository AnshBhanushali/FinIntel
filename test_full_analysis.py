import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

#############################
# Dummy Implementations
#############################

def dummy_technical_dataframe(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Return a dummy DataFrame simulating historical price data.
    We'll include a 'Close' column.
    """
    dates = pd.date_range(start_date, periods=10, freq="D")
    data = {
        "Close": np.linspace(100, 110, len(dates))
    }
    return pd.DataFrame(data, index=dates)

def dummy_sentiment_response(url: str, payload: dict):
    """
    Simulate a sentiment analysis response.
    """
    return {
        "average_sentiment": 0.85,
        "recommendation": "BUY"
    }

def dummy_llm_market_outlook(prompt: str) -> str:
    """
    Simulate an LLM (Ollama) aggregation of analysis.
    The output references technical, fundamental, and sentiment results.
    """
    return (
        "AAPL analysis: Technical indicators suggest oversold conditions. "
        "Fundamentals are robust with low debt and strong cash flow. "
        "Sentiment is highly positive. Overall recommendation: BUY."
    )

#############################
# Patching via Monkeypatch
#############################

@pytest.fixture(autouse=True)
def patch_integration(monkeypatch):
    # Patch the async function that fetches price data.
    async def async_dummy_fetch_price_data_with_cache(ticker, start, end):
        return dummy_technical_dataframe(ticker, start, end)
    monkeypatch.setattr(
        "investment_guidance.fetch_price_data_with_cache",
        async_dummy_fetch_price_data_with_cache
    )
    
    # Patch the async helper that posts JSON to the sentiment analysis agent.
    async def async_dummy_post_json(url, payload):
        return dummy_sentiment_response(url, payload)
    monkeypatch.setattr(
        "investment_guidance.post_json",
        async_dummy_post_json
    )
    
    # Patch the async helper that queries the LLM for market outlook.
    async def async_dummy_ollama_query(prompt):
        return dummy_llm_market_outlook(prompt)
    monkeypatch.setattr(
        "investment_guidance.ollama_query",
        async_dummy_ollama_query
    )
    
    # Patch the Redis client to avoid real network calls.
    class DummyRedis:
        def get(self, key):
            return None
        def setex(self, key, seconds, value):
            pass
    monkeypatch.setattr("investment_guidance.redis_client", DummyRedis())

#############################
# Integration Test
#############################

def test_unified_stock_analysis():
    """
    Simulate a full analysis for a single stock.
    
    The user provides only the ticker "AAPL" (with other parameters left to defaults).
    The aggregated response from the investment guidance endpoint should include:
      - Portfolio allocation and performance metrics.
      - A sentiment section (with a BUY recommendation).
      - A market_outlook string that includes technical, fundamental, and sentiment insights.
    """
    payload = {
        "tickers": ["AAPL"],
    }
    
    response = client.post("/investment-guidance/get-investment-guidance", json=payload)
    assert response.status_code == 200, f"Response error: {response.text}"
    
    data = response.json()
    
    # Check that the aggregated response includes key financial and performance fields.
    for key in ["allocation", "expected_annual_return", "annual_volatility", "sharpe_ratio", "sentiment", "market_outlook"]:
        assert key in data, f"Missing '{key}' in response."
    
    # Verify the market_outlook includes expected phrases.
    market_outlook = data["market_outlook"]
    assert "BUY" in market_outlook, "Market outlook should mention 'BUY' recommendation."
    assert "AAPL" in market_outlook or "technical" in market_outlook.lower(), (
        "Market outlook should reference the stock and technical analysis details."
    )
    
    # Check the sentiment field for ticker "AAPL".
    sentiment = data["sentiment"]
    assert isinstance(sentiment, dict), "Sentiment field should be a dictionary."
    assert "AAPL" in sentiment, "Sentiment should contain data for 'AAPL'."
    assert sentiment["AAPL"].get("recommendation") == "BUY", "Sentiment recommendation should be 'BUY'."
    
    # Check that the allocation is a non-empty dictionary.
    allocation = data["allocation"]
    assert isinstance(allocation, dict) and allocation, "Allocation should be a non-empty dictionary."
    
    print("Aggregated Analysis Response:")
    print(data)
