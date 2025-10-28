```{r}
# 🏡 House Prices: Advanced Regression Techniques

# =====================================================

#

# This project predicts residential property sale prices using R and modern

# machine learning regression techniques. It applies statistical preprocessing,

# feature engineering, and ensemble modeling to achieve competitive performance

# in the Kaggle competition "House Prices: Advanced Regression Techniques".

#

# -----------------------------------------------------

# 🔑 Key Highlights

# -----------------------------------------------------

#

# 1. Data Cleaning & Outlier Removal

# - Removed extreme outliers (GrLivArea > 4600).

# - Combined training and test sets for consistent preprocessing.

# - Log-transformed SalePrice to normalize its distribution.

#

# 2. Feature Engineering

# - Created domain-informed features:

# • TotalSF = GrLivArea + TotalBsmtSF

# • HouseAge = YrSold - YearBuilt

# • YearsSinceRemodel = YrSold - YearRemodAdd

# • OverallScore = OverallQual * OverallCond

# • TotalBath = BsmtFullBath + (BsmtHalfBath*0.5) + FullBath + (HalfBath*0.5)

# - Converted ordinal quality features (ExterQual, KitchenQual, etc.) to numeric scales (0–5).

#

# 3. Missing Data Imputation

# - Semantic: "None" for absent features (no garage, no pool).

# - Structural: 0 for missing numeric areas (no basement).

# - Statistical: median/mode imputation for remaining NAs.

#

# 4. Feature Transformation & Encoding

# - Applied log1p() to skewed numeric predictors (skewness > 0.7).

# - Removed near-zero variance and highly correlated variables (r > 0.95).

# - Used one-hot encoding for categorical variables.

# - Centered and scaled all predictors.

#

# 5. Modeling and Ensemble Strategy

# - Trained 3 models with 10-fold cross-validation:

# • GLMNet (ElasticNet)

# • Random Forest (Ranger)

# • XGBoost

# - Weighted Blending:

# FinalPred_log = 0.50*XGBoost + 0.30*RandomForest + 0.20*GLMNet

# SalePrice = exp(FinalPred_log)

#

# 6. Performance

# | Model         | CV RMSE  |

# |---------------|----------|

# | XGBoost       | ~0.127   |

# | Random Forest | ~0.134   |

# | GLMNet        | ~0.130   |

# | **Blend**     | **0.1267** |

#

# 7. Output

# - File: submission_blended.csv

# - Columns: Id, SalePrice

#

# -----------------------------------------------------

# 📘 Theoretical Appendix

# -----------------------------------------------------

#

# 1. Feature Engineering

# Concept: domain-informed data representation.

# Example: TotalSF = GrLivArea + TotalBsmtSF

#

# 2. Regularization (ElasticNet)

# Balances L1 (Lasso) and L2 (Ridge) penalties:

# minimize ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²

#

# 3. Ensemble Learning

# Combines complementary models to reduce variance and bias.

# Blending improves robustness and generalization.

#

# 4. Evaluation Metric (RMSE)

# RMSE = sqrt( mean( (ŷ - y)² ) )

# Measures average deviation of predicted vs actual log-prices.

#

# 5. Log Transformation

# Stabilizes variance and normalizes error distribution.

# Predictions are back-transformed using exp().

#

# 6. Feature Scaling

# Standardization (mean=0, sd=1) ensures balanced model gradients.

#

# -----------------------------------------------------

# 📊 Workflow Overview

# -----------------------------------------------------

#

# 1. Import train/test datasets.

# 2. Clean, impute, and preprocess data.

# 3. Engineer domain-relevant features.

# 4. Train GLMNet, Random Forest, and XGBoost using caret.

# 5. Blend predictions with optimized weights.

# 6. Export final submission CSV.

#

# -----------------------------------------------------

# 🧭 Insights

# -----------------------------------------------------

# - Feature quality drives model performance.

# - Regularization + blending improve generalization.

# - Semantic handling of NAs enhances model interpretability.

#

# -----------------------------------------------------

# 🚀 Future Work

# -----------------------------------------------------

# - Bayesian or genetic hyperparameter optimization.

# - Meta-stacking with ElasticNet meta-learner.

# - Geospatial feature inclusion for neighborhood context.

# - Advanced imputation using missForest or iterative KNN.

#

# -----------------------------------------------------

# 👨‍💻 Author

# -----------------------------------------------------

# Simon (2025)

# Data Science enthusiast passionate about interpretable ML,

# ensemble modeling, and reproducible feature-driven pipelines.

#

# -----------------------------------------------------

# 🧾 Usage

# -----------------------------------------------------

# Required R Packages:

# caret, xgboost, glmnet, randomForest, doParallel, dplyr, ggplot2

#

# To run:

# 1. Open `HousePrices_AdvancedRegression.R`

# 2. Execute full pipeline (data import → model blending)

# 3. Output file: submission_blended.csv

```

