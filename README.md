# On-Demand Mobility: Predictive Pricing & Optimization

## Executive Summary
This project develops a high-precision predictive framework for ride-sharing platforms (Uber/Lyft). By analyzing over 690,000 records, the project utilizes ensemble learning and **Industrial Engineering (IE)** optimization techniques to predict price fluctuations with an **$R^2$ of 0.96**.

## 🛠 Technical Methodology

### 1. Multi-Stage Feature Selection
To handle the high-dimensional nature of the dataset (50+ features), a rigorous selection pipeline was implemented:
- **LASSO Regression:** Applied for L1-regularization to shrink non-essential coefficients.
- **Recursive Feature Elimination (RFE):** Utilized to isolate the top 10 most influential predictors, significantly reducing computational noise.

### 2. Taguchi Method for Hyperparameter Optimization
Instead of a standard, resource-heavy Grid Search, this project employs a **Taguchi L9 Orthogonal Array**. 
- **Efficiency:** Optimized three XGBoost hyperparameters at three levels each using only **9 trials** instead of **27**.
- **Impact:** Achieved a **66% reduction** in experimental design space while maintaining maximum predictive precision.

**Optimization Objective ($J$):**
$$\min \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

## 📊 Performance Metrics
The optimized **XGBoost** model demonstrated superior convergence and accuracy:

| Metric | Value |
| :--- | :--- |
| **Coefficient of Determination ($R^2$)** | **0.96** |
| **Root Mean Square Error (RMSE)** | **1.64** |
| **Mean Absolute Error (MAE)** | **1.10** |

## Key IE Insights
- **Surge Multiplier & Distance:** Identified as the primary drivers of price volatility through feature importance analysis.
- **Resource Optimization:** The application of Taguchi arrays proves that large-scale ML models can be tuned with minimal computational overhead, a critical factor for real-time industrial systems.

## Repository Structure
- `On_Demand_Mobility_Pricing_Optimization.ipynb`: Full implementation pipeline.

## Note on Data
For repository portability, the full 690k record dataset is replaced with a representative sample. The complete source data can be found on Kaggle (Uber/Lyft Boston Dataset).
