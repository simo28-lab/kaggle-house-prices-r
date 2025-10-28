# House Prices: Advanced Regression Techniques (Kaggle)
This project presents a robust and highly competitive solution for the classic Kaggle Regression Challenge, "House Prices: Advanced Regression Techniques." It applies a disciplined feature-driven pipeline and sophisticated ensemble modeling to accurately predict residential property sale prices. The primary goal is to achieve minimal root mean squared error (RMSE) on the target variable.

## Key Highlights: A Modern ML Pipeline
The solution leverages modern machine learning techniques, including Lasso Feature Selection, Ordinal Encoding, and Weighted Blending of heterogeneous models (XGBoost, Random Forest, ElasticNet).

### 1. Data Cleaning & Outlier Management
This stage ensures the integrity and stability of the regression models.

Outlier Removal: Extreme, high-leverage outliers (specifically GrLivArea>4600) were removed to prevent distortion of the predictive surface.

Data Consolidation: Training and test sets were combined for consistent and unified preprocessing.

Target Normalization: The dependent variable was log-transformed (SalePrice→log(SalePrice)) to stabilize variance and normalize the error distribution.

### 2. Feature Engineering: Domain Expertise
New features were created based on domain knowledge to maximize predictive power:

TotalSF=GrLivArea+TotalBsmtSF (Total livable square footage)

HouseAge=YrSold−YearBuilt

YearsSinceRemodel=YrSold−YearRemodAdd

OverallScore=OverallQual×OverallCond

TotalBath=BsmtFullBath+(BsmtHalfBath×0.5)+FullBath+(HalfBath×0.5)

Ordinal Conversion: Categorical quality features (e.g., ExterQual, KitchenQual) were converted to an interpretable numeric scale (0 to 5).

### 3. Missing Data Imputation Strategy
A semantic approach was adopted to handle missing values based on their meaning in the real estate domain:

Semantic Imputation: $\mathbf{NA}$s were replaced with "None" (for categorical features like Alley or PoolQC) indicating absence of the item.

Structural Imputation: $\mathbf{NA}$s in numeric area fields (e.g., $\mathbf{BsmtFinSF1}$) were set to **$\mathbf{0}$**.

Statistical Imputation: Median/mode imputation was used for the small number of remaining missing values (LotFrontage, etc.).

### 4. Feature Transformation & Scaling
Skewness Correction: log 
1p
​
  transformation was applied to highly skewed numeric predictors (skewness>0.7).

Feature Filtering: Near-zero variance features were removed, and highly correlated variables (r>0.95) were pruned.

Encoding: Categorical variables were processed using One-Hot Encoding.

Scaling: All final numeric predictors were centered and scaled (mean =0, sd =1) to ensure balanced model gradients.

### 5. Modeling and Ensemble Strategy
Three high-performance models were trained using 10-fold cross-validation for reliable performance estimation:

XGBoost: Excellent for non-linear feature interactions.

Random Forest (Ranger): Provides stability and low variance.

GLMNet (ElasticNet): Combines interpretability with effective regularization.

Weighted Blending:
The final prediction is an Ensemble Weighted Blend designed to capitalize on the strengths of each model:

FinalPred 
log
​
 =0.50⋅XGBoost+0.30⋅RandomForest+0.20⋅GLMNet
The final SalePrice is obtained by back-transforming: SalePrice=exp(FinalPred 
log
​
 ).

### 6. Performance Metrics
The blend successfully improves generalization, achieving a lower RMSE than any single base learner.

Model	Cross-Validation RMSE (log(SalePrice))
Blend (Ensemble)	0.1267
XGBoost	∼0.127
GLMNet	∼0.130
Random Forest	∼0.134


### 7. Output
The final submission file: submission_blended.csv, containing columns Id and SalePrice.

## Theoretical Appendix: Key Concepts
### 1. Feature Engineering
Concept: The process of using domain knowledge to create predictive features from raw data, enhancing the model's understanding of complex relationships. Example: The creation of TotalSF aggregates two key areas into a single, highly predictive metric.

### 2. Regularization (ElasticNet)
ElasticNet combines L1 (Lasso, for feature selection) and L2 (Ridge, for coefficient stability) penalties. This approach promotes both sparsity and robustness.

β
min
​
 ∥y−Xβ∥ 
2
 +λ 
1
​
 ∥β∥ 
1
​
 +λ 
2
​
 ∥β∥ 
2
2
​
 
### 3. Ensemble Learning & Blending
Concept: Combining multiple distinct models (heterogeneous) to leverage their complementary strengths. Blending is a simple, effective form of ensemble learning that significantly improves robustness and generalization compared to single models.

### 4. Evaluation Metric (RMSE)
The Root Mean Squared Error measures the average magnitude of the errors. When applied to log(SalePrice), it penalizes large percentage errors more heavily, reflecting the typical metric used in pricing models.

RMSE= 
n
1
​
  
i=1
∑
n
​
 ( 
y
^
​
  
i
​
 −y 
i
​
 ) 
2
 

​
 
### 5. Log Transformation
Purpose: Applied to both SalePrice and various skewed predictors to better meet the assumptions of linear models (Gaussian errors) and to stabilize variance. Predictions are returned to the original scale using exp().

### 6. Feature Scaling
Standardization (mean =0, standard deviation =1) ensures that no single feature dominates the model simply due to the magnitude of its values, crucial for gradient-based models like XGBoost and penalized regression like GLMNet.

## Workflow Overview
Import train/test datasets

Clean, impute, and preprocess data

Engineer domain-relevant features

Train GLMNet, RandomForest, and XGBoost using caret

Blend predictions with optimized weights

Export final submission CSV

## Insights
Feature Quality: Rigorous feature engineering and selection (via Lasso) are the primary drivers of model performance.

Ensemble Power: Regularization and blending consistently improve generalization beyond the best single learner.

Semantic Handling: Treating missing data based on real-world meaning enhances model interpretability and accuracy.

## Future Work
Implement Meta-Stacking using ElasticNet as a meta-learner over Out-Of-Fold (OOF) predictions.

Conduct Bayesian or Genetic hyperparameter optimization instead of grid search.

Incorporate Geospatial Features for neighborhood context (e.g., distance to schools, central business district).

Employ advanced imputation techniques like missForest or iterative KNN for non-structural $\mathbf{NA}$s.
## Author
Simon (2025) Data Science enthusiast passionate about interpretable ML, ensemble modeling, and reproducible feature-driven pipelines.
