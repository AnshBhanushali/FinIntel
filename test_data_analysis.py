import pytest
import pandas as pd
import torch
from unittest.mock import patch, AsyncMock
from data_analysis import fetch_stock_data, compute_indicators, forecast_arima, forecast_prophet, forecast_lstm

# 1️⃣ Mock test for fetch_stock_data function
@pytest.mark.asyncio
async def test_fetch_stock_data():
    with patch("data_analysis.yfinance.download") as mock_yf:
        mock_yf.return_value = pd.DataFrame({"Close": [100, 102, 105]}, index=pd.date_range("2024-01-01", periods=3))

        result = await fetch_stock_data("AAPL", "2024-01-01", "2024-01-03")
        assert not result.empty
        assert "Close" in result.columns

# 2️⃣ Test compute_indicators function
def test_compute_indicators():
    df = pd.DataFrame({"Close": [100, 102, 105]}, index=pd.date_range("2024-01-01", periods=3))
    indicators = ["SMA_50", "RSI"]
    result = compute_indicators(df, indicators)

    assert "SMA_50" in result.columns
    assert "RSI" in result.columns

# 3️⃣ Mock test for forecast_arima
def test_forecast_arima():
    series = pd.Series([100, 102, 105, 110, 115], index=pd.date_range("2024-01-01", periods=5))

    result = forecast_arima(series, steps=3)

    assert isinstance(result, list)
    assert len(result) == 3

# 4️⃣ Mock test for forecast_prophet
def test_forecast_prophet():
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=5),
        "y": [100, 102, 105, 110, 115]
    })

    result = forecast_prophet(df, steps=3)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert all("ds" in entry for entry in result)

# 5️⃣ Mock test for LSTM forecast
def test_forecast_lstm():
    df = pd.DataFrame({"Close": [100, 102, 105]}, index=pd.date_range("2024-01-01", periods=3))

    # Create a valid mock state_dict
    mock_state_dict = {"fc.weight": torch.randn(1, 50), "fc.bias": torch.randn(1)}

    # Patch torch.load to return a mock state_dict
    with patch("torch.load", return_value=mock_state_dict), \
         patch("data_analysis.LSTMModel.load_state_dict") as mock_load_state, \
         patch("data_analysis.LSTMModel.eval") as mock_eval:

        mock_load_state.return_value = None  # Mock load_state_dict
        mock_eval.return_value = None  # Mock eval() so it doesn't actually run the model

        result = forecast_lstm(df, steps=3)

    assert isinstance(result, list)
    assert len(result) == 3
