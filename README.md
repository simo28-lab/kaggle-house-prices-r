House Prices: Advanced Regression Techniques — Feature-Driven Ensemble
Author: [Your Name]

Kaggle Competition: House Prices – Advanced Regression Techniques

Score: RMSE ≈ 0.1267 (Public LB)
Tools: R, caret, glmnet, xgboost, randomForest

🚀 Project Overview

This project predicts house sale prices based on a mix of structural, neighborhood, and quality-related features.
The pipeline combines rigorous data cleaning, feature engineering, and model ensembling to achieve strong performance while maintaining interpretability.

My approach aims to balance statistical transparency (GLMNet), tree-based robustness (Random Forest), and boosting efficiency (XGBoost).
The final model blends these algorithms in a weighted ensemble for improved generalization.

🧰 Technical Stack
Category	Package
Data Handling	tidyverse, dplyr, DataExplorer
Preprocessing	caret, e1071, corrplot
Modeling	glmnet, randomForest, gbm, xgboost
Parallelization	doParallel
Visualization	ggplot2, corrplot
🧼 1. Data Preparation
🏗️ Loading & Cleaning

Removed known outliers (GrLivArea > 4600) from the training data to prevent distortion.

Created a backup target variable (SalePriceRaw) for inspection and comparison.

🔎 Exploratory Data Analysis

Used DataExplorer::introduce() and histograms to explore variable distributions.

Applied log-transformation to SalePrice to normalize skewness before modeling.

🧪 2. Feature Engineering

Feature engineering was the key performance driver. The dataset was merged (train + test) for consistent preprocessing, followed by several imputation and transformation strategies:

🧩 Semantic Imputation

Categorical NAs interpreted as absence → replaced with "None".

Numeric NAs where absence = zero → replaced with 0.

Other missing values filled using median or mode imputation depending on type.

🧱 New Engineered Features
Feature	Description
TotalSF	Total living area (above + basement)
HouseAge	Years since construction
YearsSinceRemodel	Years since last remodeling
OverallScore	Interaction between overall quality and condition
TotalBath	Combined full + half baths (weighted)
⚙️ Encoded Quality Mappings

Converted ordinal quality/condition variables (ExterQual, KitchenQual, GarageQual, etc.) into numeric scales (0–5) for model interpretability.

🔢 Log Transform & Variance Filtering

Applied log1p() to highly skewed numeric features (skewness > 0.7).

Dropped near-zero variance and highly correlated (>0.95) predictors.

🔠 Encoding & Standardization

Used caret::dummyVars() for one-hot encoding of categorical features.

Scaled all numeric features via centering & standardization.

🤖 3. Modeling Pipeline

The final training matrix retained ~70 top features selected through GLMNet feature importance (non-zero coefficients).
All models were validated with 10-fold cross-validation under a unified control (trainControl(method = "cv", number = 10)).

⚡ Parallel Computing

Training leveraged multi-core processing using doParallel to speed up the CV process.

🧠 Models Used
Model	Package	Key Idea	Role
GLMNet (ElasticNet)	glmnet	Linear regularization with L1/L2 penalties	Feature selection & interpretability
Random Forest (Ranger)	randomForest	Ensemble of decision trees (bagging)	Robustness to non-linearities
XGBoost	xgboost	Gradient boosting on decision trees	High performance & fine-tuning potential
🧩 4. Ensemble Blending Strategy

After cross-validation, predictions were combined using a weighted log-space blend:

Final Prediction (log)
=
0.50
×
𝑋
𝐺
𝐵
𝑜
𝑜
𝑠
𝑡
+
0.30
×
𝑅
𝑎
𝑛
𝑑
𝑜
𝑚
𝐹
𝑜
𝑟
𝑒
𝑠
𝑡
+
0.20
×
𝐺
𝐿
𝑀
𝑁
𝑒
𝑡
Final Prediction (log)=0.50×XGBoost+0.30×RandomForest+0.20×GLMNet

Converted back to price scale via exponential transformation:

SalePrice
=
exp
⁡
(
Prediction (log)
)
SalePrice=exp(Prediction (log))

Final output:

submission_blended.csv
