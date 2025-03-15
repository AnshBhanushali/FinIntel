import logging
import datetime
import random
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger("setiment_analysis_agent")
logger.setLevel(logging.INFO)

app = FastAPI