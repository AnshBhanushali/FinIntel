import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_data_analysis():
    response = client.post(
        "/analyze-technical/", 
        json={"ticker": "AAPL", "start_date": "2023-01-01", "end_date": "2023-03-01"}
    )
    assert response.status_code == 200

def test_fundamental_analysis():
    response = client.post(
        "/fundamentals/analyze-fundamentals/",
        json={"ticker": "AAPL", "year": 2024}
    )
    assert response.status_code == 200

def test_sentiment_analysis():
    response = client.post(
        "/sentiment/analyze-sentiment/",
        json={"ticker": "AAPL", "sources": ["news", "twitter"]}
    )
    assert response.status_code == 200
