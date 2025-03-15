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