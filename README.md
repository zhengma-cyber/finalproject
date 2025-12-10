# Forecasting Room Occupancy 30 Minutes Ahead from Environmental Sensors

This repository contains the code and report for my DATA 1030 final project at Brown University.  
The goal is to **forecast room occupancy 30 minutes ahead** using historical environmental sensor readings
(temperature, light, CO₂, humidity ratio) from the UCI Occupancy Detection dataset.

- **Course:** DATA 1030 – Data Science
- **Author:** Zheng Ma
- **Repo:** https://github.com/zhengma-cyber/finalproject

---

## 1. Project Overview

Modern HVAC systems often run on fixed schedules, regardless of whether a room is actually occupied.  
This project reframes the standard occupancy detection problem into a **forecasting** task:

> Predict whether a room will be occupied (1) or empty (0) **30 minutes in the future**,  
> using only data available up to time *t − 30 minutes*.

Key ideas:

- Use a **sliding window** over historical sensor data (from *t−60* to *t−30* minutes).
- Engineer **window statistics** (mean, std) and **strided lags** every 5 minutes.
- Compare several ML models: Logistic Regression, KNN, SVM, Random Forest, and XGBoost.
- Focus on **F1-score and PR-AUC** due to class imbalance.
- Analyze **uncertainty** (splitting vs randomness) and **interpretability** (global + local).

---

## 2. Dataset

The project uses the **UCI Occupancy Detection Data Set**:

- Dataset link: <http://archive.ics.uci.edu/dataset/357/occupancy+detection>
- Original paper:  
  L. Candanedo and V. Feldheim, *Accurate occupancy detection of an office room from light, temperature, humidity and CO₂ measurements using statistical learning models*, Energy and Buildings, 2016.

Raw CSV files from UCI (e.g. `datatraining.txt`, `datatest.txt`, `datatest2.txt`) are placed in the `data/` folder and merged in the preprocessing step.

---

## 3. Repository Structure

```text
.
├── data/          # Raw and preprocessed data (not tracked in detail on GitHub)
├── figures/       # All generated figures (EDA, CV vs Test, feature importance, SHAP, etc.)
├── results/       # Saved model outputs, predictions, intermediate results
├── report/        # Final compiled PDF of the report
├── src/           # Source code (python notebooks)
├── .gitignore
├── LICENSE
└── README.md
