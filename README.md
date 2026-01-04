# California Wildfire Risk Predictor: Cost-Sensitive Anomaly Detection

Abstract
This project focuses on developing a Fire Risk Predictor, a machine learning model designed for predicting the likelihood of a wildfire based on features that include time of the year, temperature, wind speeds, and numerous more. The purpose of this model is to proactively assess / predict fire danger levels. Having this ability can lead to better improved resource allocation, and stronger levels of preparation to reduce the social, economical, and environmental impact of these fires. This project uses a large dataset of historical wildfires that have occurred in California, and will utilize / apply data preprocessing techniques, exploratory data analysis, and predictive modeling techniques that will identify patterns and hidden information from the dataset that influence the potential fire risk. 

Introduction
The goal of this project is to build a machine learning model that accurately identifies the likelihood of a wildfire on any given day in California. Wildfire instances have become increasingly more frequent in the state as time has passed, with some having extremely devastating social, economic, and environmental impacts. These risks highlight the need for proactive, data driven strategies that would assist in minimizing these potential impacts. Our motivation is to design a predictive system that can assist locations and have them better prepared. To do this our plan is to leverage historical wildfire data to better understand the conditions that have strong relationships to wildfire ignition.  

## Project Overview

This project implements a **high-recall binary classification model** to predict the daily risk of a wildfire starting in California.

Given the inherent **class imbalance** (many 'No Fire' days vs. few 'Fire' days), the methodology focuses on **Anomaly Detection** principles and utilizes **Cost-Sensitive Learning** to minimize the critical risk of a **False Negative** (missing an actual fire).

The final goal is to output an actionable **HIGH_FIRE_RISK_DAY** signal based on a probability threshold.

---

## Data Source and Preparation

### 1. Source Data
* **Dataset:** California Weather and Fire Prediction Dataset (1984–2025) with Engineered Features.
* **Target Variable:** `FIRE_START_DAY` (Binary: Yes/No).

### 2. Time-Series Filtering (Concept Drift Mitigation)
* The data is filtered to the **most recent 10 years** of available records (e.g., 2015-2025). This addresses **concept drift**—the change in fire dynamics, land use, and climate over decades.

### 3. Training and Testing Split
* The data is split strictly **chronologically** to prevent future information from leaking into the training process.
    * **Training Set:** The earliest $\approx 80\%$ of the 10-year data. (Used for training with weighting).
    * **Test Set:** The most recent $\approx 20\%$ of the data. (Reserved for final, unbiased performance evaluation).

---

## Methodology: Cost-Sensitive Learning

To overcome the severe class imbalance, we employ **Class Weighting** during model training, which is preferred over SMOTE to preserve the time-series integrity of the engineered features (like lagged precipitation).

### 1. Weight Calculation
The **scale\_pos\_weight** parameter assigns a higher penalty to errors made on the minority class.


### 2. Impact on Training
The calculated weight is applied to the model's loss function. This makes a **False Negative** (missing a fire) highly expensive, forcing the model to prioritize **Recall** and aggressively seek out fire-conducive patterns.

### 3. Final Output Target
The model's probability output will be converted into a binary signal using a safety-optimized threshold ($\tau$):

* If $P(\text{Fire}) \geq \tau$, the final output is **`HIGH_RISK_FIRE_DAY`**.
* If $P(\text{Fire}) < \tau$, the final output is **`NORMAL_DAY`**.
---
