// app/page.tsx
"use client";
import { useState } from "react";
import StockInput from "../components/StockInput";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";

// Define the type for the submit payload
interface StockSubmitPayload {
  tickers: string[];
  riskTolerance: string;
  totalCapital: number;
}

const HomePage = () => {
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleStockSubmit = async ({ tickers, riskTolerance, totalCapital }: StockSubmitPayload) => {
    setLoading(true);
    setError("");
    setAnalysis(null);

    const payload = {
      tickers,
      risk_tolerance: riskTolerance,
      initial_capital: totalCapital,
    };

    try {
      const res = await fetch(
        "http://localhost:8000/investment-guidance/get-investment-guidance",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      if (!res.ok) {
        throw new Error(`HTTP error! Status: ${res.status}`);
      }
      const data = await res.json();
      setAnalysis(data);
    } catch (err: any) {
      console.error("Error fetching analysis:", err);
      setError(err.message || "Failed to fetch analysis. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Simulated LLM-like explanation generator
  const explainMetric = (metric: string, value: any, context: any = {}) => {
    switch (metric) {
      case "tickers":
        if (!Array.isArray(value) || !value) {
          return "No stocks selected yet. This section will list the companies you’ve chosen once data is available.";
        }
        return `You’ve selected ${value.join(", ")} for your portfolio. Each of these is a stock ticker, a short code for a company listed on the stock market. For example, AAPL is Apple Inc. This list tells us which companies’ performance we’re analyzing to see how your money could grow.`;

      case "risk_tolerance":
        return value === "low"
          ? `Your risk tolerance is set to "low." In technical terms, this means you prefer investments with smaller price swings, even if it might mean lower returns. Think of it like choosing a calm boat ride over a rollercoaster—you’re prioritizing stability over big thrills.`
          : value === "high"
          ? `Your risk tolerance is "high." This means you’re comfortable with investments that can have big ups and downs, technically called high volatility, because they might give you higher returns. It’s like betting on a racehorse—risky, but the payoff could be huge.`
          : `Your risk tolerance is "medium." This is a balanced approach, mixing some stable investments with riskier ones. Technically, it’s about optimizing your "risk-reward ratio"—you’re aiming for decent returns without too much wild price action.`;

      case "total_capital":
        return `Your total capital is $${value.toLocaleString()}. This is the amount you’re putting into your portfolio, like the fuel for your investment engine. In technical terms, it’s your "initial investment," and we use it to calculate how much to allocate to each stock based on your risk and goals.`;

      case "expected_annual_return":
        return `Your expected annual return is ${value}%. This predicts how much your portfolio might grow each year over time. For example, with $${context.total_capital}, you could see it grow by $${(context.total_capital * value / 100).toLocaleString()} in year one, compounding over multiple years—though markets fluctuate, so it’s an estimate.`;

      case "annual_volatility":
        return `Your annual volatility is ${value}%. This shows how much your portfolio’s value might swing each year. Technically, it’s the "standard deviation" of returns—over years, a ${value}% volatility could mean swings of $${(context.total_capital * value / 100).toLocaleString()} up or down annually, reflecting risk over time.`;

      case "sharpe_ratio":
        return value >= 1.5
          ? `Your Sharpe Ratio is ${value}, which is excellent! This measures return per unit of risk over time. Above 1.5 means you’re getting strong returns for the risk, like a high-efficiency engine running smoothly across years.`
          : value < 1
          ? `Your Sharpe Ratio is ${value}, a bit low. This compares returns to risk, and below 1 suggests more risk than reward over time—like a car that guzzles gas for short trips. Consider adjusting your strategy.`
          : `Your Sharpe Ratio is ${value}, solid! It balances return and risk over years. Around 1 means a fair deal—like a reliable car rental. Higher is better, but this works well.`;

      case "sentiment":
        return `The sentiment analysis shows: ${JSON.stringify(value)}. This uses NLP to gauge public mood about your stocks from news or social media. Positive (e.g., 0.7) suggests optimism that could lift prices over time; negative (e.g., -0.2) might drag them down.`;

      case "market_outlook":
        return `The market outlook says: "${value}" This forecasts market trends over time using data and models—bullish means up, bearish means down, neutral is steady. It’s like a multi-year weather prediction for your investments.`;

      case "allocation":
        return `Your portfolio allocation is ${JSON.stringify(value)}. This splits your $${context.total_capital} across stocks, optimized for risk and return over years—like budgeting for a long-term plan. Each stock’s share aims for growth and stability.`;

      case "stock_prices":
        return `These are price histories for your stocks over time. Each line tracks value changes—upward trends show growth, dips show losses. Technically, it’s a time series, giving you a multi-year view of past performance to predict future moves.`;

      default:
        return "No explanation available for this metric yet.";
    }
  };

  // Simulated multi-year data generators
  const generateReturnProjection = (initialCapital: number, annualReturn: number) => {
    const years = 5;
    const data = [];
    let capital = initialCapital;
    for (let i = 0; i <= years; i++) {
      data.push({
        year: `Year ${i}`,
        return: i === 0 ? 0 : (capital * annualReturn) / 100,
        cumulative: capital - initialCapital,
      });
      capital += (capital * annualReturn) / 100; // Compound annually
    }
    return data;
  };

  const generateVolatilityRange = (volatility: number) => {
    return [
      { year: "Avg", volatility },
      { year: "Low", volatility: volatility * 0.8 },
      { year: "High", volatility: volatility * 1.2 },
    ];
  };

  const generateSharpeComparison = (sharpe: number) => {
    return [
      { year: "Year 1", portfolio: sharpe, benchmark: 1.0 },
      { year: "Year 2", portfolio: sharpe * 1.05, benchmark: 1.0 },
      { year: "Year 3", portfolio: sharpe * 0.95, benchmark: 1.0 },
    ];
  };

  const generatePriceHistory = (ticker: string) => {
    const today = new Date();
    return Array.from({ length: 36 }, (_, i) => ({
      date: new Date(today.getFullYear() - 2, i, 1).toISOString().split("T")[0], // 3 years of monthly data
      price: 100 + Math.random() * 50 + i * 2, // Simulated upward trend
    }));
  };

  return (
    <main className="main-container">
      <style>{`
        .main-container {
          padding: 2rem;
          font-family: 'Arial', sans-serif;
          max-width: 1200px;
          margin: 0 auto;
        }
        .main-title {
          color: #333;
          text-align: center;
          margin-bottom: 2rem;
        }
        .loading {
          text-align: center;
          color: #007BFF;
        }
        .error {
          text-align: center;
          color: #ff4444;
        }
        .analysis-container {
          margin-top: 2rem;
          padding: 1.5rem;
          background: #f9f9f9;
          border-radius: 8px;
        }
        .metric-section {
          margin-bottom: 2rem;
          padding: 1rem;
          background: white;
          border-radius: 4px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-section h3 {
          color: #333;
          margin-bottom: 1rem;
        }
        .explanation {
          color: #666;
          font-size: 0.95rem;
          line-height: 1.5;
          margin-top: 0.5rem;
        }
        .preformatted {
          background: #fff;
          padding: 1rem;
          border-radius: 4px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          overflow-x: auto;
          margin: 1rem 0;
        }
        .stock-chart {
          margin-top: 1rem;
        }
        .stock-chart h4 {
          color: #555;
          margin-bottom: 0.5rem;
        }
      `}</style>
      <h1 className="main-title">Financial Dashboard</h1>
      <StockInput onSubmit={handleStockSubmit} />
      {loading && <p className="loading">Loading analysis...</p>}
      {error && <p className="error">{error}</p>}
      {analysis && (
        <div className="analysis-container">
          <h2>Aggregated Portfolio Analysis</h2>

          <div className="metric-section">
            <h3>Selected Stocks</h3>
            <p><strong>Value:</strong> {analysis.tickers?.join(", ") || "Not available"}</p>
            <p className="explanation">{explainMetric("tickers", analysis.tickers)}</p>
          </div>

          <div className="metric-section">
            <h3>Risk Tolerance</h3>
            <p><strong>Value:</strong> {analysis.risk_tolerance}</p>
            <p className="explanation">{explainMetric("risk_tolerance", analysis.risk_tolerance)}</p>
          </div>

          <div className="metric-section">
            <h3>Total Capital</h3>
            <p><strong>Value:</strong> ${analysis.total_capital?.toLocaleString()}</p>
            <p className="explanation">{explainMetric("total_capital", analysis.total_capital)}</p>
          </div>

          <div className="metric-section">
            <h3>Expected Annual Return (5-Year Projection)</h3>
            <p><strong>Value:</strong> {analysis.expected_annual_return}%</p>
            <p className="explanation">{explainMetric("expected_annual_return", analysis.expected_annual_return, { total_capital: analysis.total_capital })}</p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={generateReturnProjection(analysis.total_capital, analysis.expected_annual_return)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis label={{ value: "Return ($)", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="return" stroke="#82ca9d" name="Annual Return" />
                <Line type="monotone" dataKey="cumulative" stroke="#ff7300" name="Cumulative Growth" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="metric-section">
            <h3>Annual Volatility (Range)</h3>
            <p><strong>Value:</strong> {analysis.annual_volatility}%</p>
            <p className="explanation">{explainMetric("annual_volatility", analysis.annual_volatility, { total_capital: analysis.total_capital })}</p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={generateVolatilityRange(analysis.annual_volatility)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis label={{ value: "Volatility (%)", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="volatility" fill="#8884d8" name="Volatility Range" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="metric-section">
            <h3>Sharpe Ratio (3-Year Comparison)</h3>
            <p><strong>Value:</strong> {analysis.sharpe_ratio}</p>
            <p className="explanation">{explainMetric("sharpe_ratio", analysis.sharpe_ratio)}</p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={generateSharpeComparison(analysis.sharpe_ratio)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis label={{ value: "Ratio", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="portfolio" stroke="#ffc658" name="Portfolio Sharpe" />
                <Line type="monotone" dataKey="benchmark" stroke="#8884d8" name="Benchmark (1.0)" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="metric-section">
            <h3>Sentiment Analysis</h3>
            <pre className="preformatted">{JSON.stringify(analysis.sentiment, null, 2)}</pre>
            <p className="explanation">{explainMetric("sentiment", analysis.sentiment)}</p>
            {analysis.sentiment && (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={
                    analysis.sentiment && typeof analysis.sentiment === "object" && !Array.isArray(analysis.sentiment)
                      ? Object.entries(analysis.sentiment)
                          .filter(([_, value]) => typeof value === "number")
                          .map(([key, value]) => ({ name: key, score: value as number }))
                      : Array.isArray(analysis.sentiment)
                      ? analysis.sentiment.map((item: any) => ({
                          name: item.ticker || "Sentiment",
                          score: item.score || 0,
                        }))
                      : []
                  }
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis label={{ value: "Sentiment Score", angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="score" fill="#ff7300" name="Sentiment Score" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="metric-section">
            <h3>Market Outlook</h3>
            <p>{analysis.market_outlook}</p>
            <p className="explanation">{explainMetric("market_outlook", analysis.market_outlook)}</p>
          </div>

          <div className="metric-section">
            <h3>Portfolio Allocation</h3>
            <pre className="preformatted">{JSON.stringify(analysis.allocation, null, 2)}</pre>
            <p className="explanation">{explainMetric("allocation", analysis.allocation, { total_capital: analysis.total_capital })}</p>
            {analysis.allocation && (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={Object.entries(analysis.allocation).map(([ticker, percent]) => ({ ticker, percent }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="ticker" />
                  <YAxis label={{ value: "Allocation (%)", angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="percent" fill="#00C49F" name="Allocation" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {analysis.stock_prices && (
            <div className="metric-section">
              <h3>Stock Price History (3 Years)</h3>
              <p className="explanation">{explainMetric("stock_prices", analysis.stock_prices)}</p>
              {analysis.stock_prices.map((stock: any, index: number) => (
                <div key={index} className="stock-chart">
                  <h4>{stock.ticker}</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={stock.data || generatePriceHistory(stock.ticker)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis label={{ value: "Price ($)", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="price" stroke="#8884d8" name={`${stock.ticker} Price`} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </main>
  );
};

export default HomePage;