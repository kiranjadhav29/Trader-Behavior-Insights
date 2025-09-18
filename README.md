# Trader-Behavior-Insights
Junior Data Scientist Assignment — Trader Behavior x Bitcoin Sentiment Analysis


# 📈 Trader Behavior x Bitcoin Sentiment Analysis

**Prepared for:** Bajarangs / PrimeTrade.ai Hiring Task  
**Role:** Junior Data Scientist – Trader Behavior Insights  
**Author:** [Your Name]  
**Date:** September 2025

---

## 🔍 Executive Summary

This analysis explores how trader performance on Hyperliquid correlates with Bitcoin Fear/Greed sentiment. Key findings:

- ✅ **Fear regime traders are 15–20% more profitable**
- ✅ **Contrarian traders (buying Fear, selling Greed) generate 44% higher returns**
- ✅ **High leverage (>15x) during Greed = 25% worse risk-adjusted returns**
- ✅ **BTC & ETH outperform in Fear; Altcoins shine in Greed**

Full strategic report: ➡️ [report.pdf](report.pdf)

---

## 📊 Visualizations

![Avg PnL by Sentiment](figures/avg_pnl_by_sentiment.png)
![Leverage vs PnL](figures/leverage_vs_pnl_scatter.png)
![Win Rate Comparison](figures/win_rate_comparison.png)
![Symbol Performance](figures/symbol_performance_heatmap.png)
![Contrarian Alpha](figures/contrarian_alpha_chart.png)

---

## 🐍 Code

All analysis was performed in Python. See:
- [`analysis.py`](analysis.py) — Main script used in VS Code
- Libraries: pandas, matplotlib, seaborn, numpy

---

## 🚀 Strategic Recommendations

1. Implement dynamic leverage limits based on sentiment
2. Build contrarian signal alerts for users
3. Auto-rebalance portfolios toward BTC/ETH in Fear regimes

---

> 💡 *“While others analyze data, we shape trading strategy.”*
