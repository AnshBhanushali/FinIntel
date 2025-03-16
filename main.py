from fastapi import FastAPI
from data_analysis import app as data_analysis_app
from fundamental_analysis import app as fundamental_analysis_app
from sentiment_analysis import app as sentiment_analysis_app
from investment_guidance import app as investment_guidance_app  

app = FastAPI(title="Unified AI Agents API")

@app.get("/")
async def root():
    return {"message": "Welcome to the Unified AI Agents API. Visit /docs for API documentation."}

# Mount your sub‑apps
app.mount("/analyze-technical", data_analysis_app)
app.mount("/fundamentals", fundamental_analysis_app)
app.mount("/sentiment", sentiment_analysis_app)
app.mount("/investment-guidance", investment_guidance_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
