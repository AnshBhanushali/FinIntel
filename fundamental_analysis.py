import logging
import os
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sec_edgar_downloader import Downloader
import pandas as pd

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

@app.post("/analyze-fundamentals/")
def analyze_fundamentals(request: FundamentalRequest):
    try:
        # Step 1: Download the 10-K (or 10-Q) filing from SEC EDGAR
        dl = Downloader(os.path.join("tmp_edgar", request.ticker))
        filings = dl.get("10-K", request.ticker, amount=1, 
                         after=f"{request.year}-01-01", before=f"{request.year}-12-31")

        if not filings:  # If no 10-K found for that year
            raise HTTPException(status_code=404, detail="No 10-K found for that year.")

        # The downloaded file(s) are stored in tmp_edgar/<ticker>/10-K/<folder>/...
        filing_text = ""
        for doc in filings:
            with open(doc, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            filing_text += text

        # Step 2: Extract relevant sections using regex
        business_pattern = re.compile(
            r"item\s1\.\s+business(.*?)item\s1a\.\s+risk\s+factors",
            re.IGNORECASE | re.DOTALL
        )
        risk_pattern = re.compile(
            r"item\s1a\.\s+risk\s+factors(.*?)item\s1b\.\s+unresolved\s+staff\s+comments",
            re.IGNORECASE | re.DOTALL
        )

        business_match = business_pattern.search(filing_text)
        risk_match = risk_pattern.search(filing_text)

        business_summary_text = business_match.group(1) if business_match else "N/A"
        risk_factors_text = risk_match.group(1) if risk_match else "N/A"

        # Step 3: Summarize extracted text using T5
        summarized_business = (
            summarizer(business_summary_text, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
            if business_summary_text != "N/A" else "N/A"
        )
        summarized_risks = (
            summarizer(risk_factors_text, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
            if risk_factors_text != "N/A" else "N/A"
        )

        # Hardcoded fundamental metrics (in a real scenario, these would be calculated/extracted)
        pe_ratio = 17.5
        de_ratio = 0.45
        roi = 0.12
        cash_flow = 5.6e9

        # Step 4: Advanced ML approach to compute a fundamental score.
        # Attempt to load a pre-trained model (e.g., a regression model trained on historical fundamentals)
        try:
            import pickle
            with open("models/fundamentals_model.pkl", "rb") as f:
                ml_model = pickle.load(f)
            # Construct feature vector; this should match your ML model's expected input.
            features = [pe_ratio, de_ratio, roi, cash_flow]
            advanced_score = ml_model.predict([features])[0]
        except Exception as ml_e:
            logger.error(f"ML model loading/prediction failed: {ml_e}")
            # Fallback: use sentiment analysis on the summarized texts to adjust a base score.
            business_sentiment = sentiment_analyzer(summarized_business)[0] if summarized_business != "N/A" else {"label": "NEUTRAL", "score": 0.5}
            risk_sentiment = sentiment_analyzer(summarized_risks)[0] if summarized_risks != "N/A" else {"label": "NEUTRAL", "score": 0.5}

            # For business, a positive sentiment is desirable; for risk factors, a negative sentiment might be expected.
            business_factor = business_sentiment["score"] if business_sentiment["label"] == "POSITIVE" else (1 - business_sentiment["score"])
            risk_factor = (1 - risk_sentiment["score"]) if risk_sentiment["label"] == "NEGATIVE" else risk_sentiment["score"]

            # Compute a heuristic advanced score (this is a placeholder for a real ML model)
            advanced_score = (business_factor * roi * 100) / (pe_ratio * de_ratio * (risk_factor + 0.1))

        # Step 5: Use an LLM (via Llama) for advanced insights.
        llama_prompt = (
            f"Analyze the following summarized information and fundamental metrics for a company:\n"
            f"Business Summary: {summarized_business}\n"
            f"Risk Factors Summary: {summarized_risks}\n"
            f"Financial Metrics: P/E Ratio: {pe_ratio}, D/E Ratio: {de_ratio}, ROI: {roi}, Cash Flow: {cash_flow}\n"
            f"Based on this information, provide an advanced analysis and potential investment strategy."
        )
        llama_insight = llama_query(llama_prompt)

        # Return all results
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
