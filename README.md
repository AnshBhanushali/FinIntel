# 📈 AI-Driven Multi-Agent Financial Decision System  

## 🔥 Overview  

This project is an **AI-powered multi-agent system** for **financial decision-making and stock market analysis**. It integrates **real-time stock data**, **technical and fundamental analysis**, **sentiment analysis**, and **AI-driven investment strategies** to provide users with **data-backed financial insights**.

Users can **input a stock ticker** and receive **detailed reports**, including **price predictions, market trends, risk analysis, and portfolio recommendations**.

---

## ⚡ Key Features  

### 🏦 **1. Stock Market Data Analysis Agent**  
- Fetches **real-time and historical** stock data from sources like **Yahoo Finance, Alpha Vantage, or Bloomberg API**.  
- Computes **technical indicators** (SMA, EMA, RSI, MACD, Bollinger Bands) for **trend analysis**.  
- Uses **time-series forecasting models** like **ARIMA, LSTM, or GARCH** to predict **price movements**.  
- Outputs **interactive financial charts** showing **growth trends** over **days, months, and years**.

### 🏢 **2. Fundamental Analysis & Company Insights Agent**  
- Scrapes **SEC financial reports (10-K, 10-Q, earnings reports, balance sheets, income statements)**.  
- Extracts key **financial metrics** (P/E ratio, Debt-to-Equity, ROI, Cash Flow).  
- Uses **NLP and Knowledge Graphs** to **summarize** business models, future plans, and **risk factors**.  
- Tracks **institutional investor holdings (13F), insider trading (Form 4 SEC), and Google Trends data**.

### 📊 **3. Sentiment Analysis & Predictive Modeling Agent**  
- Aggregates **financial sentiment** from **Bloomberg, CNBC, Twitter, Reddit** using **Transformer-based NLP models (FinBERT, GPT-4, T5)**.  
- Performs **sentiment classification** to assess market **optimism vs. pessimism**.  
- Integrates **Monte Carlo Simulation & VaR (Value at Risk)** for **stock risk modeling**.  
- Categorizes stocks into **Short-term, Long-term, or Speculative** investments.  
- Provides **Buy/Sell/Hold** recommendations with **confidence scores**.

### 📈 **4. AI-Driven Investment Guidance Agent**  
- Aggregates **insights** from all **agents** to create an **investment strategy**.  
- Uses **Modern Portfolio Theory (MPT) and Sharpe Ratio** to suggest **portfolio allocations**.  
- Provides **scenario-based analysis** for **inflation, interest rates, and economic shifts**.  
- Displays **real-time market data, sentiment trends, and AI-driven forecasts** in a **dashboard** (Streamlit/Dash).  
- Sends **alerts** for **price movements, earnings calls, and regulatory filings**.

---

## 🚀 Tech Stack  

### **🔧 Backend:**  
- **Python** with **FastAPI** (REST API & async processing).  
- **AI/ML Models:** FinBERT, GPT-4, T5, LSTM, ARIMA, Random Forest, XGBoost.  
- **Financial Data APIs:** Alpha Vantage, Yahoo Finance, Bloomberg, SEC Filings.  

### **💻 Frontend:**  
- **Next.js / React** for UI/UX.  
- **Streamlit / Dash** for **interactive financial dashboards**.  

### **🗄 Database:**  
- **PostgreSQL / MySQL** for **structured financial data**.  
- **MongoDB** for **unstructured news/sentiment data**.  

---

## 📌 Installation & Setup  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/anshbhanushali/AI-Financial-Decision-System.git
cd AI-Financial-Decision-System
```

### **2️⃣ Setup the Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate      # Windows
```

### **3️⃣ Install Backend Dependencies**  
```bash
pip install -r requirements.txt
```

### **4️⃣ Run FastAPI Server**  
```bash
uvicorn app.main:app --reload
```
API will be available at **http://127.0.0.1:8000**.

### **5️⃣ Install Frontend Dependencies & Start Next.js Server**  
```bash
cd frontend
npm install
npm run dev
```
Frontend will be available at **http://localhost:3000**.

---

## 🔬 API Endpoints  

| Method | Endpoint | Description |
|--------|----------|------------|
| **POST** | `/analyze-technical/` | Fetch stock market data, compute indicators, and forecast trends. |
| **POST** | `/analyze-fundamentals/` | Fetch financial reports, extract metrics, and summarize insights. |
| **POST** | `/analyze-sentiment/` | Perform sentiment analysis on news & social media data. |
| **POST** | `/generate-investment-strategy/` | Generate an AI-driven investment strategy based on data analysis. |

---

## 🛠 Additional Enhancements  

✔️ **Explainable AI (XAI):** Uses **SHAP/LIME** to explain AI decisions.  
✔️ **Multi-Model Ensemble Learning:** Combines **Random Forest, LSTMs, GPT models** for accuracy.  
✔️ **Backtesting & Simulation:** Allows **historical strategy testing**.  
✔️ **Regulatory Compliance:** Follows **SEC & FINRA** guidelines.  

---

## 🤝 Contributing  

1. **Fork** the repo.  
2. Create a **feature branch** (`git checkout -b feature-name`).  
3. **Commit** your changes (`git commit -m "Add new feature"`).  
4. **Push** to the branch (`git push origin feature-name`).  
5. Open a **Pull Request**.  

---

## 📜 License  

This project is licensed under the **MIT License**.  

---
