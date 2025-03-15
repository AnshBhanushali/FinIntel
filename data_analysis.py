import logging
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import aiohttp
import pandas as pd
import pandas_ta as ta

# Options to use statsmodels, prophet, and your custom LSTM
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import torch
from torch import nn

# Setup logger
logger = logging.getLogger("data_analysis_agent")
logger.setLevel(logging.INFO)

class StockRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    indicators: Optional[List[str]] = ["SMA_50", "EMA_20", "RSI", "MACD", "BBANDS"]
    forecast_models: Optional[List[str]] = ["ARIMA", "PROPHET"]  

app = FastAPI

CACHE = {}

class LSTMModel(nn.Modeule):
    """ this for advanced forcasting """
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

async def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """This would asynchronously fetch data stck data from yfinance"""

    # Here we check cache first
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cache_key in CACHE:
        logger.info(f"cache hit for {cache_key}")
        return CACHE[cache_key].copy()

    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date} ...")
    # Example: using yfinance asynchronously 
    async with aiohttp.ClientSession() as session:
        pass

    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise HTTPException(status_code=404, detail="No data found for specified ticker/date range.")
    
    # Cache result
    CACHE[cache_key] = data
    return data.copy()

def compute_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """Compute selected technical indicators on the price DataFrame for advanced stock reading"""
    df = df.copy()
    for ind in indicators:
        if ind.startswith("SMA"):
            length = int(ind.split("_")[1])
            df[ind] = ta.sma(df['Close'], length=length)
        elif ind.startswith("EMA"):
            length = int(ind.split("_")[1])
            df[ind] = ta.ema(df['Close'], length=length)
        elif ind == "RSI":
            df[ind] = ta.rsi(df['Close'])
        elif ind == "MACD":
            macd_values = ta.macd(df['Close'])
            df['MACD_line'] = macd_values['MACD_12_26_9']
            df['MACD_signal'] = macd_values['MACDs_12_26_9']
            df['MACD_hist'] = macd_values['MACDh_12_26_9']
        elif ind == "BBANDS":
            bb = ta.bbands(df['Close'])
            df["BB_upper"] = bb['BBU_20_2.0']
            df["BB_middle"] = bb['BBM_20_2.0']
            df["BB_lower"] = bb['BBL_20_2.0']
    return df

def forecast_arima(series: pd.Series, steps: int = 30):
    try:
        model = ARIMA(series, order = (1,1,1))
        model_fit = model.fit()
        pred = model_fit.forecast(steps = steps)
        return pred.tolist()
    except Exception as e:
        logger.error(f"ARIMA forecasting failed: {e}")
        return []

def forecast_prophet(df: pd.DataFrame, steps: int = 30):
    """
    Expects df with columns ['ds', 'y']. Convert from your original DF: ds=Date, y=Close
    """
    try:
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps).to_dict(orient='records')
    except Exception as e:
        logger.error(f"Prophet forecasting failed: {e}")
        return []

def forecast_lstm(df: pd.DataFrame, steps: int = 30):
    """ LSTM approach. Here we train our model offline or dynamically """
    try:
        # Hypothetical pre-trained model path
        model_path = "models/lstm_stock_model.pt"  
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # This function would return a list of predicted prices or a timeseries
        return [100.0 + i for i in range(steps)]  
    
    except Exception as e:
        logger.error(f"LSTM forecasting failed: {e}")
        return []

def ensemble_forecast(df: pd.DataFrame, models: List[str]) -> dict:
    """
    Combine forecasts from ARIMA, Prophet, LSTM, etc.
    A simple ensemble could be the average of predictions.
    """
    close_series = df['Close'].dropna()
    results = {}

    if "ARIMA" in models:
        results["ARIMA"] = forecast_arima(close_series)
    if "PROPHET" in models:
        # Need to rename columns for Prophet
        prophet_df = pd.DataFrame({
            "ds": close_series.index,
            "y": close_series.values
        })
        results["PROPHET"] = forecast_prophet(prophet_df)
    if "LSTM" in models:
        results["LSTM"] = forecast_lstm(close_series)

    combined_forecast = []
    try:
        num_models = len([k for k in results if results[k]])
        if num_models > 0:
            # Just a naive approach
            forecast_length = 30 
            for step_idx in range(forecast_length):
                step_values = []
                for m_key, f_data in results.items():
                    if m_key == "PROPHET":
                        # get yhat from the step
                        step_values.append(f_data[step_idx]['yhat'])
                    else:
                        step_values.append(f_data[step_idx])
                combined_forecast.append(sum(step_values) / len(step_values))
        else:
            combined_forecast = []
    except Exception as e:
        logger.error(f"Ensemble forecast failed: {e}")

    return {
        "individual": results,
        "ensemble": combined_forecast
    }

@app.post("/analyze-technical/")
async def analyze_technical(request: StockRequest):
    try:
        df = await fetch_stock_data(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date
        )
        # Compute indicators
        df_with_ind = compute_indicators(df, request.indicators)

        # Forecast
        ensemble_results = ensemble_forecast(df_with_ind, request.forecast_models)

        # Package results
        return {
            "ticker": request.ticker,
            "indicators": request.indicators,
            "latest_indicators": df_with_ind.iloc[-1][request.indicators].to_dict(),
            "forecasts": ensemble_results
        }
    except Exception as e:
        logger.error(f"analyze_technical endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))