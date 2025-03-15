import logging
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import aiohttp
import pandas as pd
import pandas_ta as ta

from pypfopt import EfficientFrontier, risk_models, expected_returns

logger = logging.getLogger("investment_guidance_agent")
logger.setLevel(logging.INFO)

app = FastAPI

class GuidanceRequest(BaseModel):
    tickers: List[str]
    initial_capital: float = 10000
    risk_tolerance: Optional[str] = "medium"