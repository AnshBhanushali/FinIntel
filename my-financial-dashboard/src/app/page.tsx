// app/page.tsx
"use client";
import { useState } from "react";
import StockInput from "../components/StockInput";

export default function HomePage() {
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleStockSubmit = async (ticker: string) => {
    setLoading(true);
    setError("");
    setAnalysis(null);

    const payload = { tickers: [ticker] };

    try {
      // Update this URL if your FastAPI backend is hosted elsewhere.
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
    <main style={{ padding: "2rem", fontFamily: "Arial, sans-serif" }}>
      <h1>Financial Dashboard</h1>
      <StockInput onSubmit={handleStockSubmit} />
      {loading && <p>Loading analysis...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
      {analysis && (
        <div style={{ marginTop: "2rem" }}>
          <h2>Aggregated Analysis for {analysis.ticker}</h2>
          <p>
            <strong>Risk Tolerance:</strong> {analysis.risk_tolerance}
          </p>
          <p>
            <strong>Total Capital:</strong> {analysis.total_capital}
          </p>
          <p>
            <strong>Expected Annual Return:</strong>{" "}
            {analysis.expected_annual_return}
          </p>
          <p>
            <strong>Annual Volatility:</strong> {analysis.annual_volatility}
          </p>
          <p>
            <strong>Sharpe Ratio:</strong> {analysis.sharpe_ratio}
          </p>
          <h3>Sentiment Analysis:</h3>
          <pre>{JSON.stringify(analysis.sentiment, null, 2)}</pre>
          <h3>Market Outlook:</h3>
          <p>{analysis.market_outlook}</p>
          <h3>Portfolio Allocation:</h3>
          <pre>{JSON.stringify(analysis.allocation, null, 2)}</pre>
        </div>
      )}
    </main>
  );
}
