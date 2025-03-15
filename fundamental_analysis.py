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

logger = logging.getLogger("fundamental_analysis_agent")
logger.setLevel(logging.INFO)

app = FastAPI()

# Summarization model
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

class FundamentalRequest(BaseModel):
    ticker: str
    year: Optional[int] = 2024

@app.post("/analyze-fundamentals/")
def analyze_fundamentals(request: FundamentalRequest):
    try:
        # Step 1: Download the 10-K (or 10-Q) from SEC EDGAR
        dl = Downloader(os.path.join("tmp_edgar", request.ticker))
        filings = dl.get("10-K", request.ticker, amount=1, after=f"{request.year}-01-01", before=f"{request.year}-12-31")

        if not filings:  # If no 10-K found for that year
            raise HTTPException(status_code=404, detail="No 10-K found for that year.")

        # The downloaded file is in tmp_edgar/<ticker>/10-K/<some_folder>/...
        # Step 2: Parse the HTML/text to extract relevant sections
        # For example, let's locate 'Item 1. Business' or 'Risk Factors'
        filing_text = ""
        for doc in filings:
            with open(doc, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            filing_text += text

        business_pattern = re.compile(r"item\s1\.\s+business(.*?)item\s1a\.\s+risk\s+factors", re.IGNORECASE|re.DOTALL)
        risk_pattern = re.compile(r"item\s1a\.\s+risk\s+factors(.*?)item\s1b\.\s+unresolved\s+staff\s+comments", re.IGNORECASE|re.DOTALL)

        business_summary = business_pattern.search(filing_text)
        risk_factors = risk_pattern.search(filing_text)