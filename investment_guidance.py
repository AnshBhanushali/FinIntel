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