from fastapi import FastAPI
from data_analysis import app as data_analysis_app
from fundamental_analysis import app as fundamental_analysis_app
from sentiment_analysis import app as sentiment_analysis_app

app = FastAPI(title="Unified AI Agents API")

@app.get("/")
async def root():
    return {"message": "Welcome to the Unified AI Agents API. Visit /docs for API documentation."}

# Option 1: If your sub‑apps define their endpoints at "/", then mount them as follows:
app.mount("/analyze-technical", data_analysis_app)         # Endpoint becomes POST /analyze-technical/
app.mount("/fundamentals", fundamental_analysis_app)         # Endpoint becomes POST /fundamentals/
app.mount("/sentiment", sentiment_analysis_app)
     # Endpoint becomes POST /analyze-sentiment/

# Option 2: If you prefer the endpoints to have a longer path (as your tests expect),
# adjust the mount paths accordingly. For example:
# app.mount("/analyze-technical", data_analysis_app)
# app.mount("/fundamentals/analyze-fundamentals", fundamental_analysis_app)
# app.mount("/sentiment/analyze-sentiment", sentiment_analysis_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
