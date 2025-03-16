import logging
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import aiohttp
import pandas as pd
import pandas_ta as ta
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import torch
from torch import nn

# Setup logger
logger = logging.getLogger("data_analysis_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False  # Prevent duplicate logs

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "UMWT45DZO156S1PZ"  # Your provided key

class StockRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    indicators: Optional[List[str]] = ["SMA_50", "EMA_20", "RSI", "BBANDS"]
    forecast_models: Optional[List[str]] = ["ARIMA", "PROPHET"]

app = FastAPI()

CACHE = {}

class LSTMModel(nn.Module):
    """This is for advanced forecasting using LSTM."""
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

async def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch stock data asynchronously from Alpha Vantage."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cache_key in CACHE:
        logger.info(f"Cache hit for {cache_key}")
        return CACHE[cache_key].copy()

    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date} ...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Alpha Vantage API returned status {response.status}")
                raise HTTPException(status_code=502, detail="Failed to fetch data from Alpha Vantage")
            data = await response.json()
            logger.info(f"API response keys: {list(data.keys())}")  # Log response structure
            logger.debug(f"Full API response: {data}")

    if "Time Series (Daily)" not in data:
        error_msg = data.get("Note", data.get("Error Message", "Unknown error"))
        logger.error(f"No data found for {ticker}: {error_msg}")
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}: {error_msg}")

    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[df.index.to_series().between(start_date, end_date)]
    
    if df.empty:
        logger.error(f"No data available for {ticker} between {start_date} and {end_date}")
        raise HTTPException(status_code=404, detail="No data found for specified ticker/date range")

    logger.info(f"Data fetched successfully, shape: {df.shape}")
    CACHE[cache_key] = df
    return df.copy()

def compute_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """Compute selected technical indicators on the price DataFrame."""
    df = df.copy()
    logger.debug(f"Computing indicators on DataFrame with shape {df.shape}, columns: {df.columns.tolist()}")
    
    for ind in indicators:
        try:
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
                bb = ta.bbands(df['Close'], length=20, std=2)
                if bb is not None and not bb.empty:
                    df["BB_upper"] = bb.get('BBU_20_2.0', bb.get('BBU_5_2.0'))
                    df["BB_middle"] = bb.get('BBM_20_2.0', bb.get('BBM_5_2.0'))
                    df["BB_lower"] = bb.get('BBL_20_2.0', bb.get('BBL_5_2.0'))
                else:
                    logger.warning("Bollinger Bands calculation returned None or empty")
        except Exception as e:
            logger.error(f"Error computing indicator {ind}: {e}")
            raise
    
    logger.debug(f"Indicators computed, new columns: {df.columns.tolist()}")
    return df

def forecast_arima(series: pd.Series, steps: int = 30):
    try:
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=steps)
        return pred.tolist()
    except Exception as e:
        logger.error(f"ARIMA forecasting failed: {e}")
        return []

def forecast_prophet(df: pd.DataFrame, steps: int = 30):
    """Expects a DataFrame with columns ['ds', 'y'] (ds=Date, y=Close)."""
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
    """LSTM approach. Placeholder - assumes a pre-trained model."""
    try:
        model_path = "models/lstm_stock_model.pt"
        model = LSTMModel()
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
        model.eval()
        return [df['Close'].iloc[-1] + i * 0.1 for i in range(steps)]
    except Exception as e:
        logger.error(f"LSTM forecasting failed: {e}")
        return []

def ensemble_forecast(df: pd.DataFrame, models: List[str]) -> dict:
    """Combine forecasts from ARIMA, Prophet, LSTM, etc."""
    close_series = df['Close'].dropna()
    results = {}

    if "ARIMA" in models:
        results["ARIMA"] = forecast_arima(close_series)
    if "PROPHET" in models:
        prophet_df = pd.DataFrame({"ds": close_series.index, "y": close_series.values})
        results["PROPHET"] = forecast_prophet(prophet_df)
    if "LSTM" in models:
        results["LSTM"] = forecast_lstm(df)

    combined_forecast = []
    try:
        num_models = len([k for k in results if results[k]])
        if num_models > 0:
            forecast_length = 30
            for step_idx in range(forecast_length):
                step_values = []
                for m_key, f_data in results.items():
                    if m_key == "PROPHET":
                        step_values.append(f_data[step_idx]['yhat'])
                    else:
                        step_values.append(f_data[step_idx])
                combined_forecast.append(sum(step_values) / len(step_values))
    except Exception as e:
        logger.error(f"Ensemble forecast failed: {e}")

    return {"individual": results, "ensemble": combined_forecast}

async def ollama_query(prompt: str) -> str:
    """Query Ollama for advanced AI insights."""
    url = "http://localhost:11434/run"
    payload = {"model": "llama2", "prompt": prompt}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Ollama query failed with status {response.status}")
                    return "Ollama query failed."
                data = await response.json()
                return data.get("output", "No output from Ollama.")
    except Exception as e:
        logger.error(f"Ollama query exception: {e}")
        return "Ollama query encountered an exception."

@app.post("/")
async def analyze_technical(request: StockRequest):
    try:
        df = await fetch_stock_data(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date
        )
        df_with_ind = compute_indicators(df, request.indicators)
        ensemble_results = ensemble_forecast(df_with_ind, request.forecast_models)

        available_indicators = [ind for ind in request.indicators if ind in df_with_ind.columns]
        latest_indicators = df_with_ind.iloc[-1][available_indicators].dropna().to_dict() if available_indicators else {}

        prompt = (
            f"Provide an advanced analysis for the following stock data:\n"
            f"Ticker: {request.ticker}\n"
            f"Indicators: {available_indicators}\n"
            f"Latest Indicator values: {latest_indicators}\n"
            f"Forecasts: {ensemble_results}\n"
            f"Generate insights and potential trading strategies."
        )

        ai_insight = await ollama_query(prompt)

        return {
            "ticker": request.ticker,
            "indicators": request.indicators,
            "available_indicators": available_indicators,
            "latest_indicators": latest_indicators,
            "forecasts": ensemble_results,
            "ai_insight": ai_insight
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"analyze_technical endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)