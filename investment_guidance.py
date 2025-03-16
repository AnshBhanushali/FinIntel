import logging
import asyncio
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from pypfopt import EfficientFrontier, risk_models, expected_returns

logger = logging.getLogger("investment_guidance")
logger.setLevel(logger.INFO)

app = FastAPI()

class GuidanceRequest(BaseModel):
    tickers : List[str]
    initial_capital: float = 10000
    risk_tolerance: Optional[str] = "medium" # User can choose between low, medium, high

