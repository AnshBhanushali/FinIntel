"use client";
import { useState } from "react";
import StockInput from "../components/StockInput";

// Define the interface for submit function parameters
interface StockSubmitParams {
  tickers: string[];
  riskTolerance: string;
  totalCapital: number;
}

// Regular component (not the default export)
function HomePage() {
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Typed function parameter destructuring
  const handleStockSubmit = async ({
    tickers,
    riskTolerance,
    totalCapital,
  }: StockSubmitParams) => {
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

  return (
    <main className="main-container">
      <h1 className="main-title">Financial Dashboard</h1>

      <StockInput onSubmit={handleStockSubmit} />

      {loading && <p className="loading">Loading analysis...</p>}
      {error && <p className="error">{error}</p>}

      {analysis && (
        <div className="analysis-container">
          <h2>Aggregated Portfolio Analysis</h2>

          <div className="metrics-grid">
            <div className="metric-card">
              <strong>Tickers:</strong> {analysis.tickers?.join(", ")}
            </div>
            <div className="metric-card">
              <strong>Risk Tolerance:</strong> {analysis.risk_tolerance}
            </div>
            <div className="metric-card">
              <strong>Total Capital:</strong> $
              {analysis.total_capital?.toLocaleString()}
            </div>
            <div className="metric-card">
              <strong>Expected Annual Return:</strong>{" "}
              {analysis.expected_annual_return}%
            </div>
            <div className="metric-card">
              <strong>Annual Volatility:</strong> {analysis.annual_volatility}%
            </div>
            <div className="metric-card">
              <strong>Sharpe Ratio:</strong> {analysis.sharpe_ratio}
            </div>
          </div>

          <h3>Sentiment Analysis</h3>
          <pre className="preformatted">
            {JSON.stringify(analysis.sentiment, null, 2)}
          </pre>

          <h3>Market Outlook</h3>
          <p>{analysis.market_outlook}</p>

          <h3>Portfolio Allocation</h3>
          <pre className="preformatted">
            {JSON.stringify(analysis.allocation, null, 2)}
          </pre>
        </div>
      )}
    </main>
  );
}

// Inline styles (can be moved to a CSS file)
const styles = `
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
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }
  .metric-card {
    padding: 1rem;
    background: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  .preformatted {
    background: #fff;
    padding: 1rem;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    overflow-x: auto;
  }
`;

// Default export wrapper to include inline styles
export default function HomePageWithStyles() {
  return (
    <>
      <style>{styles}</style>
      <HomePage />
    </>
  );
}
