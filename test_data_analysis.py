import pytest
import pandas as pd
import torch
from unittest.mock import patch, AsyncMock
from data_analysis import (
    fetch_stock_data,
    compute_indicators,
    forecast_arima,
    forecast_prophet,
    forecast_lstm,
)

# Define a fake response class that supports async context management.
class FakeResponse:
    def __init__(self, json_data, status=200):
        self._json_data = json_data
        self.status = status

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

# 1️⃣ Test for fetch_stock_data using aiohttp.ClientSession.get patched with an AsyncMock.
@pytest.mark.asyncio
async def test_fetch_stock_data():
    # Prepare a fake JSON response similar to what Alpha Vantage would return.
    fake_json = {
        "Time Series (Daily)": {
            "2024-01-01": {
                "1. open": "100",
                "2. high": "102",
                "3. low": "99",
                "4. close": "100",
                "5. volume": "1000000",
            },
            "2024-01-02": {
                "1. open": "100",
                "2. high": "103",
                "3. low": "100",
                "4. close": "102",
                "5. volume": "1200000",
            },
            "2024-01-03": {
                "1. open": "102",
                "2. high": "106",
                "3. low": "101",
                "4. close": "105",
                "5. volume": "1500000",
            },
        }
    }
    
    # Create a FakeResponse instance.
    fake_response = FakeResponse(fake_json, status=200)
    # Create an AsyncMock that returns our FakeResponse.
    fake_get = AsyncMock(return_value=fake_response)
    
    # Patch the aiohttp.ClientSession.get method in the data_analysis module.
    with patch("data_analysis.aiohttp.ClientSession.get", new=fake_get):
        result = await fetch_stock_data("AAPL", "2024-01-01", "2024-01-03")
        assert not result.empty
        assert "Close" in result.columns

# 2️⃣ Test for compute_indicators function.
def test_compute_indicators():
    df = pd.DataFrame({"Close": [100, 102, 105]}, index=pd.date_range("2024-01-01", periods=3))
    indicators = ["SMA_50", "RSI"]
    result = compute_indicators(df, indicators)
    assert "SMA_50" in result.columns
    assert "RSI" in result.columns

# 3️⃣ Test for forecast_arima.
def test_forecast_arima():
    series = pd.Series([100, 102, 105, 110, 115], index=pd.date_range("2024-01-01", periods=5))
    result = forecast_arima(series, steps=3)
    assert isinstance(result, list)
    assert len(result) == 3

# 4️⃣ Test for forecast_prophet.
def test_forecast_prophet():
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=5),
        "y": [100, 102, 105, 110, 115]
    })
    result = forecast_prophet(df, steps=3)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all("ds" in entry for entry in result)

# 5️⃣ Test for forecast_lstm.
def test_forecast_lstm():
    df = pd.DataFrame({"Close": [100, 102, 105]}, index=pd.date_range("2024-01-01", periods=3))
    # Create a valid mock state_dict.
    mock_state_dict = {"fc.weight": torch.randn(1, 50), "fc.bias": torch.randn(1)}

    # Patch torch.load, LSTMModel.load_state_dict, and LSTMModel.eval.
    with patch("torch.load", return_value=mock_state_dict), \
         patch("data_analysis.LSTMModel.load_state_dict") as mock_load_state, \
         patch("data_analysis.LSTMModel.eval") as mock_eval:
        mock_load_state.return_value = None  # Stub load_state_dict.
        mock_eval.return_value = None         # Stub eval().
        result = forecast_lstm(df, steps=3)
    assert isinstance(result, list)
    assert len(result) == 3
