import logging
import os
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import yfinance as yf  # New dependency for fundamentals

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Setup logger
logger = logging.getLogger("fundamental_analysis_agent")
logger.setLevel(logging.INFO)

app = FastAPI()

# Summarization model
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Sentiment analysis pipeline (for our ML fallback)
sentiment_analyzer = pipeline("sentiment-analysis")

class FundamentalRequest(BaseModel):
    ticker: str
    year: Optional[int] = 2024

def llama_query(prompt: str) -> str:
    """
    Query a local Llama (or similar) LLM service for advanced insights.
    Adjust the URL and model parameters as needed.
    """
    url = "http://localhost:11434/run"  # Update with your Llama endpoint URL
    payload = {
        "model": "llama2",  # Specify the model to use
        "prompt": prompt
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("output", "No output from Llama.")
        else:
            return f"Llama query failed with status code {response.status_code}"
    except Exception as e:
        return f"Llama query exception: {e}"

def get_fundamentals_from_yfinance(ticker: str):
    """
    Use yfinance to fetch fundamental metrics.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    # Use the stock info to get basic metrics. Fallback to hardcoded values if unavailable.
    pe_ratio = info.get("trailingPE") or 17.5
    de_ratio = info.get("debtToEquity")
    if de_ratio is None:
        de_ratio = 0.45
    roi = info.get("returnOnAssets")
    if roi is None:
        roi = 0.12
    # For cash flow, attempt to fetch from the cashflow statement; fallback if unavailable.
    cash_flow_df = stock.cashflow
    if cash_flow_df is not None and not cash_flow_df.empty:
        # Use the most recent cash flow value (adjust column/index as needed)
        cash_flow = cash_flow_df.iloc[0, 0]
    else:
        cash_flow = 5.6e9
    return pe_ratio, de_ratio, roi, cash_flow

@app.post("/analyze-fundamentals/")
def analyze_fundamentals(request: FundamentalRequest):
    try:
        # Step 1: Fetch fundamental metrics using yfinance
        pe_ratio, de_ratio, roi, cash_flow = get_fundamentals_from_yfinance(request.ticker)
        
        # Fetch additional information (e.g., business summary) from yfinance
        stock = yf.Ticker(request.ticker)
        business_summary_text = stock.info.get("longBusinessSummary", "N/A")
        # For risk factors, we can use a placeholder or later integrate another source.
        risk_factors_text = "Risk factors not available via yfinance."
        
        # Step 2: Summarize the business summary and risk factors using T5
        summarized_business = (
            summarizer(business_summary_text, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
            if business_summary_text != "N/A" else "N/A"
        )
        summarized_risks = (
            summarizer(risk_factors_text, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
            if risk_factors_text != "N/A" else "N/A"
        )
        
        # Step 3: Advanced ML approach to compute a fundamental score.
        try:
            import pickle
            with open("models/fundamentals_model.pkl", "rb") as f:
                ml_model = pickle.load(f)
            features = [pe_ratio, de_ratio, roi, cash_flow]
            advanced_score = ml_model.predict([features])[0]
        except Exception as ml_e:
            logger.error(f"ML model loading/prediction failed: {ml_e}")
            # Fallback: use sentiment analysis on the summarized texts to adjust a base score.
            business_sentiment = sentiment_analyzer(summarized_business)[0] if summarized_business != "N/A" else {"label": "NEUTRAL", "score": 0.5}
            risk_sentiment = sentiment_analyzer(summarized_risks)[0] if summarized_risks != "N/A" else {"label": "NEUTRAL", "score": 0.5}
            business_factor = business_sentiment["score"] if business_sentiment["label"] == "POSITIVE" else (1 - business_sentiment["score"])
            risk_factor = (1 - risk_sentiment["score"]) if risk_sentiment["label"] == "NEGATIVE" else risk_sentiment["score"]
            advanced_score = (business_factor * roi * 100) / (pe_ratio * de_ratio * (risk_factor + 0.1))
        
        # Step 4: Use an LLM (via Llama) for advanced insights.
        llama_prompt = (
            f"Analyze the following summarized information and fundamental metrics for a company:\n"
            f"Business Summary: {summarized_business}\n"
            f"Risk Factors Summary: {summarized_risks}\n"
            f"Financial Metrics: P/E Ratio: {pe_ratio}, D/E Ratio: {de_ratio}, ROI: {roi}, Cash Flow: {cash_flow}\n"
            f"Based on this information, provide an advanced analysis and potential investment strategy."
        )
        llama_insight = llama_query(llama_prompt)
        
        # Step 5: Return the results
        return {
            "ticker": request.ticker,
            "pe_ratio": pe_ratio,
            "de_ratio": de_ratio,
            "roi": roi,
            "cash_flow": cash_flow,
            "business_summary": summarized_business,
            "risk_factors": summarized_risks,
            "advanced_fundamental_score": advanced_score,
            "llama_insight": llama_insight
        }
    
    except Exception as e:
        logger.error(f"Error analyzing fundamentals for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
