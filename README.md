# House Prices: Advanced Regression Techniques

This project predicts residential property sale prices using R and modern machine learning regression techniques.
It applies statistical preprocessing, feature engineering, and ensemble modeling to achieve competitive performance in the Kaggle competition “House Prices: Advanced Regression Techniques.”

# Key Highlights
## 1. Data Cleaning & Outlier Removal

Removed extreme outliers (GrLivArea > 4600)

Combined training and test sets for consistent preprocessing

Log-transformed SalePrice to normalize its distribution

## 2. Feature Engineering

Created domain-informed features:

TotalSF = GrLivArea + TotalBsmtSF

HouseAge = YrSold - YearBuilt

YearsSinceRemodel = YrSold - YearRemodAdd

OverallScore = OverallQual * OverallCond

TotalBath = BsmtFullBath + (BsmtHalfBath*0.5) + FullBath + (HalfBath*0.5)
Converted ordinal quality features (e.g., ExterQual, KitchenQual) to numeric scales (0–5).

## 3. Missing Data Imputation

Semantic: “None” for absent features (e.g., no garage, no pool)

Structural: 0 for missing numeric areas (no basement)

Statistical: Median/mode imputation for remaining NAs

## 4. Feature Transformation & Encoding

Applied log1p() to skewed numeric predictors (skewness > 0.7)

Removed near-zero variance and highly correlated variables (r > 0.95)

One-hot encoded categorical variables

Centered and scaled all predictors

## 5. Modeling and Ensemble Strategy

Trained 3 models with 10-fold cross-validation:

GLMNet (ElasticNet)

Random Forest (Ranger)

XGBoost

Weighted Blending:

FinalPred_log = 0.50*XGBoost + 0.30*RandomForest + 0.20*GLMNet
SalePrice = exp(FinalPred_log)

## 6. Performance
Model	CV RMSE
XGBoost	~0.127
Random Forest	~0.134
GLMNet	~0.130
Blend	0.1267
7. Output

File: submission_blended.csv

Columns: Id, SalePrice

# Theoretical Appendix
## 1. Feature Engineering

Concept: Domain-informed data representation.
Example: TotalSF = GrLivArea + TotalBsmtSF

## 2. Regularization (ElasticNet)

Balances L1 (Lasso) and L2 (Ridge) penalties:

min
⁡
∣
∣
𝑦
−
𝑋
𝛽
∣
∣
2
+
𝜆
1
∣
∣
𝛽
∣
∣
1
+
𝜆
2
∣
∣
𝛽
∣
∣
2
min∣∣y−Xβ∣∣
2
+λ
1
	​

∣∣β∣∣
1
	​

+λ
2
	​

∣∣β∣∣
2

## 3. Ensemble Learning

Combines complementary models to reduce variance and bias.
Blending improves robustness and generalization.

## 4. Evaluation Metric (RMSE)

𝑅
𝑀
𝑆
𝐸
=
𝑚
𝑒
𝑎
𝑛
(
(
𝑦
^
−
𝑦
)
2
)
RMSE=
mean((
y
^
	​

−y)
2
)
	​


Measures average deviation of predicted vs. actual log-prices.

## 5. Log Transformation

Stabilizes variance and normalizes error distribution.
Predictions are back-transformed using exp().

## 6. Feature Scaling

Standardization (mean = 0, sd = 1) ensures balanced model gradients.

# Workflow Overview

Import train/test datasets

Clean, impute, and preprocess data

Engineer domain-relevant features

Train GLMNet, Random Forest, and XGBoost using caret

Blend predictions with optimized weights

Export final submission CSV

# Insights

Feature quality drives model performance

Regularization + blending improve generalization

Semantic handling of NAs enhances interpretability

# Future Work

Bayesian or genetic hyperparameter optimization

Meta-stacking with ElasticNet meta-learner

Geospatial features for neighborhood context

Advanced imputation using missForest or iterative KNN

# Author

Simon (2025)
Data Science enthusiast passionate about interpretable ML,
ensemble modeling, and reproducible feature-driven pipelines.

# Usage

Required R Packages:
caret, xgboost, glmnet, randomForest, doParallel, dplyr, ggplot2

Run the pipeline:

# Step 1: Open the script
HousePrices_AdvancedRegression.R

# Step 2: Execute the full pipeline (data import → model blending)

# Step 3: Output file
submission_blended.csv
