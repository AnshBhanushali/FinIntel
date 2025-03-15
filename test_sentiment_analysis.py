import torch
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sentiment_analysis import app, query_ollama, classify_stock

client = TestClient(app)

mock_request = {
    "ticker": "AAPL",
    "sources": ["news", "twitter"]
}

mock_texts = [
    "AAPL is expected to rise significantly this quarter.",
    "Concerns about global supply chain issues affecting AAPL.",
]

mock_ollama_response = "Apple Inc. has received a generally positive outlook in the latest financial news. Sentiment Score: 0.72."

mock_sentiment_score = 0.72
mock_var_loss_low = 4.5  # Low risk (BUY)
mock_var_loss_high = 12.0  # High risk (SELL)

@pytest.fixture
def mock_finbert():
    """Mock FinBERT sentiment model with correct tensor output."""
    with patch("sentiment_analysis.finbert_model") as mock_model:
        mock_output = MagicMock()
        
        # Ensure logits return a PyTorch Tensor instead of a list
        mock_output.logits = torch.tensor([[0.1, 0.2, 1.5]])  # Fake logits: Positive sentiment
        mock_model.return_value = mock_output
        yield mock_model

@pytest.fixture
def mock_ollama():
    """Mock Ollama LLM API call."""
    with patch("sentiment_analysis.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"output": mock_ollama_response}
        yield mock_post

def test_analyze_sentiment_buy(mock_finbert, mock_ollama):
    """Test BUY recommendation with positive sentiment and low risk."""
    with patch("sentiment_analysis.simulate_monte_carlo", return_value=[105] * 1000), \
         patch("sentiment_analysis.calculate_var", return_value=mock_var_loss_low):
    
        response = client.post("/analyze-sentiment/", json=mock_request)
        assert response.status_code == 200
        data = response.json()

        assert data["ticker"] == "AAPL"
        assert data["average_sentiment"] == pytest.approx(mock_sentiment_score, rel=1e-2)
        assert data["var_loss_95"] == mock_var_loss_low
        assert data["recommendation"] == "BUY"

def test_analyze_sentiment_sell(mock_finbert):
    with patch("sentiment_analysis.requests.post") as mock_post, \
         patch("sentiment_analysis.simulate_monte_carlo", return_value=[90] * 1000), \
         patch("sentiment_analysis.calculate_var", return_value=12.0):
        
        mock_post.return_value.status_code = 200
        # Force negative sentiment by returning a low sentiment score
        mock_post.return_value.json.return_value = {"output": "Sentiment Score: 0.35"}
        
        response = client.post("/analyze-sentiment/", json=mock_request)
        assert response.status_code == 200
        data = response.json()
        assert data["recommendation"] == "SELL"


def test_analyze_sentiment_hold(mock_finbert, mock_ollama):
    """Test HOLD recommendation with neutral sentiment and moderate risk."""
    with patch("sentiment_analysis.simulate_monte_carlo", return_value=[98] * 1000), \
         patch("sentiment_analysis.calculate_var", return_value=8.0):

        response = client.post("/analyze-sentiment/", json=mock_request)
        assert response.status_code == 200
        data = response.json()

        assert data["recommendation"] == "HOLD"
