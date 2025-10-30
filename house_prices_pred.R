

# --- Setup & Packages -------------------------------------------------------
install.packages("glmnet")
install.packages("randomForest")
install.packages("gbm")
install.packages("corrplot")
install.packages("DataExplorer")
install.packages("xgboost")
library(tidyverse)
library(caret)
library(glmnet)
library(randomForest)
library(gbm)
library(corrplot)
library(DataExplorer)
library(xgboost)
library(dplyr)
set.seed(2025)

# --- Data Loading -----------------------------------------------------------

train <- read.csv("C:/Users/simon/OneDrive/Documenti/kaggle project/train.csv")
test  <- read.csv("C:/Users/simon/OneDrive/Documenti/kaggle project/test.csv")
length(train)
length(test)
head(train)


#remove known outliers of the dataset
train <- train %>% 
  filter(GrLivArea < 4600)


# Backup original 
train$SalePriceRaw <- train$SalePrice
dim(train)
# --- Exploratory Data Analysis ----------------------------------------

# Quick overview
introduce(train)
introduce(test)
# Distribution of SalePrice
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram SalePrice")

# Log-transform target (to normalize distribution)
train$LogSalePrice <- log(train$SalePrice)
train$SalePriceRaw <- train$SalePrice



# --- FEATURE ENGINEERING & ADVANCED PREPROCESSING ---------------------------

# 1. Combine Train and Test 
train_len <- nrow(train)
full_data <- bind_rows(train %>% select(-SalePrice, -SalePriceRaw, -LogSalePrice), 
                       test)
dim(full_data)

train_target <- train$LogSalePrice
train_ID <- train$Id
test_ID <- test$Id

# Remove ID from predictors
full_data <- full_data %>% select(-Id)


# 2. Semantic Imputation (Substitute NA with 0 or 'None' where NA means Absence)

# categorical variables (NA -> "None")
cols_na_to_none <- c('Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                     'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                     'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature')

for (col in cols_na_to_none) {
  full_data[[col]][is.na(full_data[[col]])] <- "None"
}

# numerical variables (NA -> 0)
cols_na_to_zero <- c('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                     'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea', 
                     'MasVnrArea')
for (col in cols_na_to_zero) {
  full_data[[col]][is.na(full_data[[col]])] <- 0
}

# 3. Statistical Imputation for real missing values (Mode/Median)

#  MSSubClass as category
full_data$MSSubClass <- as.factor(full_data$MSSubClass)


# LotFrontage (numeric) with median
full_data$LotFrontage[is.na(full_data$LotFrontage)] <- median(full_data$LotFrontage, na.rm = TRUE)

# GarageYrBlt: if NA, means "No Garage" (0/None).
full_data$GarageYrBlt[is.na(full_data$GarageYrBlt)] <- 0

# MasVnrType (categorical): 'None' most common.
full_data$MasVnrType[is.na(full_data$MasVnrType)] <- "None" 

# Electrical, MSZoning, Utilities, Exterior*, KitchenQual, SaleType, Functional 
# Imputation with the mode
cols_na_to_mode <- c('Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 
                     'Exterior2nd', 'KitchenQual', 'SaleType', 'Functional')
for (col in cols_na_to_mode) {
  mode_val <- names(sort(table(full_data[[col]]), decreasing = TRUE)[1])
  full_data[[col]][is.na(full_data[[col]])] <- mode_val
}


# 4. Feature Engineering
full_data <- full_data %>%
  mutate(
    # Total Area  (up and down)
    TotalSF = GrLivArea + TotalBsmtSF,
    
    #age of the house
    HouseAge = YrSold - YearBuilt,
    YearsSinceRemodel = YrSold - YearRemodAdd,
    
    # Overall Score
    OverallScore = OverallQual * OverallCond,
    
    # total bathrooms
    TotalBath = BsmtFullBath + (BsmtHalfBath * 0.5) + FullBath + (HalfBath * 0.5)
  ) %>%
  # remove single old features
  select(-TotalBsmtSF, -GrLivArea, -BsmtFullBath, -BsmtHalfBath, -FullBath, -HalfBath,
         -YearBuilt, -YearRemodAdd) 


# 5. (Conversion Quality/Condition in Numbers)
quality_map <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
bsmt_map <- c('None' = 0, 'Unf' = 1, 'LwQ' = 2, 'Rec' = 3, 'BLQ' = 4, 'ALQ' = 5, 'GLQ' = 6)
bsmtexposure_map <- c('None' = 0, 'No' = 1, 'Mn' = 2, 'Av' = 3, 'Gd' = 4)

full_data <- full_data %>%
  mutate(
    # standard quality (0-5)
    ExterQual = as.numeric(recode(ExterQual, !!!quality_map)),
    ExterCond = as.numeric(recode(ExterCond, !!!quality_map)),
    HeatingQC = as.numeric(recode(HeatingQC, !!!quality_map)),
    KitchenQual = as.numeric(recode(KitchenQual, !!!quality_map)),
    FireplaceQu = as.numeric(recode(FireplaceQu, !!!quality_map)),
    GarageQual = as.numeric(recode(GarageQual, !!!quality_map)),
    GarageCond = as.numeric(recode(GarageCond, !!!quality_map)),
    PoolQC = as.numeric(recode(PoolQC, !!!quality_map)),
    
    # Map Basement (0-5)
    BsmtQual = as.numeric(recode(BsmtQual, !!!quality_map)),
    BsmtCond = as.numeric(recode(BsmtCond, !!!quality_map)),
    
    # Map Finish Basement (0-6)
    BsmtFinType1 = as.numeric(recode(BsmtFinType1, !!!bsmt_map)),
    BsmtFinType2 = as.numeric(recode(BsmtFinType2, !!!bsmt_map)),
    
    # Map Exposure Basement (0-4)
    BsmtExposure = as.numeric(recode(BsmtExposure, !!!bsmtexposure_map))
  ) %>%
  # month and year as factor
  mutate(MoSold = as.factor(MoSold), YrSold = as.factor(YrSold))


# 6. logaritmic transormation of Skewed variables (Numeric)
numeric_cols <- names(full_data %>% select(where(is.numeric)))
skewed_vars <- names(full_data[numeric_cols])


# cut-off of 0.7 for skewness 
library(e1071) #skewness

for (col in numeric_cols) {
  if (skewness(full_data[[col]]) > 0.7) {
    full_data[[col]] <- log1p(full_data[[col]])
  }
}

# 7. Remove low Variance 
nzv <- nearZeroVar(full_data, saveMetrics = TRUE)
cols_to_drop_nzv <- rownames(nzv[nzv$nzv, ])
if (length(cols_to_drop_nzv) > 0) {
  cat("low variance Variables removed:", cols_to_drop_nzv, "\n")
  full_data <- full_data %>% select(-all_of(cols_to_drop_nzv))
}

# 8. Separation Train/Test and One-Hot Encoding
train_data_final <- full_data[1:train_len, ]
test_data_final <- full_data[(train_len + 1):nrow(full_data), ]

# all categorical variables left
categorical_names <- names(train_data_final %>% select(where(is.character) | where(is.factor)))
numeric_names <- names(train_data_final %>% select(where(is.numeric)))

# One-Hot Encoding with caret
dummies <- dummyVars(~ ., data = train_data_final[, categorical_names], fullRank = TRUE)
train_dummies <- predict(dummies, newdata = train_data_final[, categorical_names])
test_dummies  <- predict(dummies, newdata = test_data_final[, categorical_names])

# Combine numerical + dummy
train_final <- cbind(train_data_final[, numeric_names], train_dummies)
test_final  <- cbind(test_data_final[, numeric_names], test_dummies)


# 9.fix columns train/test
# Necessary becayse the test set could not have all training's categories and vice versa.
common_cols <- intersect(colnames(train_final), colnames(test_final))
train_final <- train_final[, common_cols]
test_final  <- test_final[, common_cols]

# 10.1 Remove Columns with SD=0 after the Scaling
zero_sd_cols_final <- names(train_final)[apply(train_final, 2, sd, na.rm=TRUE) == 0]
if (length(zero_sd_cols_final) > 0) {
 
  train_final <- train_final %>% select(-all_of(zero_sd_cols_final))
  
}

# 10.2 Remove variables highly correlated 
corr_matrix <- cor(train_final, use = "pairwise.complete.obs") # Questa riga ora non dovrebbe più dare warning

high_corr <- findCorrelation(corr_matrix, cutoff = 0.95, names = TRUE) # Aumentato a 0.95 per non perdere troppa info
if (length(high_corr) > 0) {
  cat("Variabili altamente correlate rimosse:", high_corr, "\n")
  train_final <- train_final %>% select(-all_of(high_corr))
  test_final  <- test_final %>% select(-all_of(high_corr))
}


# 11. Standardize (Scaling)
preproc <- preProcess(train_final, method = c("center", "scale"))
train_final <- predict(preproc, train_final)
test_final  <- predict(preproc, test_final)

cat("preprocessing completed","\n")
cat("Dimension Training Set (Predictors):", dim(train_final), "\n")
cat("Dimension Test Set (Predictors):", dim(test_final), "\n")




# 1. Cross-Validation (CV)
ctrl <- trainControl(method = "cv", number = 10)

# --- 2. GLMNet (ElasticNet) for Feature Importance ---
glm_model <- train(
  x = train_final, y = train_target,
  method = "glmnet",
  trControl = ctrl,
  tuneLength = 5
)

# extract variables' importance (coefficients !=0)
glm_imp <- varImp(glm_model)$importance
glm_imp_df <- data.frame(
  Feature = rownames(glm_imp),
  Overall = glm_imp$Overall
)
glm_imp_df <- glm_imp_df[order(glm_imp_df$Overall, decreasing = TRUE), ]

# remove not importante features (Lasso selection)
glm_imp_df <- glm_imp_df %>% filter(Overall > 0)

# --- 3. Reduce Dataset ---
N_FEATURES_TO_KEEP <- min(70, nrow(glm_imp_df))
final_features_selected <- head(glm_imp_df$Feature, N_FEATURES_TO_KEEP)


train_final_selected <- train_final[, final_features_selected]
test_final_selected <- test_final[, final_features_selected]

cat("\n GLMNet completed. Features selected are (N=", ncol(train_final_selected), "): \n")
print(final_features_selected)
cat("\n------------------------------------------------\n")




# --- Parallelism ---
library(e1071) 
library(doParallel) 

# --- 1. CONFIGURATION ---
cores <- detectCores() - 1
cl <- makePSOCKcluster(cores)
registerDoParallel(cl)
cat("\nInizio Modeling...\n")
cat("Parallelismo attivo su", cores, "core. Questo accelera l'addestramento.\n")

# Cross-Validation (CV) unificato
ctrl <- trainControl(method = "cv", number = 10)


# --- 2. Random Forest (Method 'ranger' faster) ---
cat("\n[1/3] Training Random Forest (Ranger) on dataset reduced (N=", ncol(train_final_selected), ")...\n")
rf_model_ranger <- train(
  x = train_final_selected,
  y = train_target,
  method = "ranger", 
  trControl = ctrl,
  tuneLength = 3, 
  importance = 'impurity'
)
rf_rmse <- min(rf_model_ranger$results$RMSE)
cat("RF (Ranger) completed. best RMSE:", rf_rmse, "\n")


# --- 3. XGBoost ---
cat("\n[2/3] Training XGBoost using Caret to obtain RMSE CV...\n")

# max_depth=4, eta=0.05, nrounds=1000 fixed parameters
xgb_fixed_grid <- expand.grid(
  nrounds = 1000, 
  max_depth = 4, 
  eta = 0.05, 
  gamma = 0,
  colsample_bytree = 0.75,
  min_child_weight = 1,
  subsample = 0.75
)

# Sostitute block xgb.train with caret::train(method = "xgbTree")
xgb_model <- train(
  x = train_final_selected,
  y = train_target,
  method = "xgbTree", # Caret
  trControl = ctrl, # Use Cross-Validation defined
  tuneGrid = xgb_fixed_grid, # train with fixed parameters
  verbose = FALSE
)
xgb_rmse <- min(xgb_model$results$RMSE)
cat("XGBoost completed. Best RMSE:", xgb_rmse, "\n")


# --- 4. GLMNet (ElasticNet) ---
cat("\n[3/3] Training GLMNet (ElasticNet) final...\n")
glm_model_final <- train(
  x = train_final_selected, y = train_target, 
  method = "glmnet",
  trControl = ctrl,
  tuneLength = 5
)
glm_rmse <- min(glm_model_final$results$RMSE)
cat("GLMNet completed. Best RMSE:", glm_rmse, "\n")


# --- 5. STOP CLUSTER ---
stopCluster(cl)
registerDoSEQ()

cat("\n------------------------------------------------\n")
cat("Final RMSE of Cross-Validation:\n")
cat("1. XGBoost: ", round(xgb_rmse, 5), "\n")
cat("2. GLMNet: ", round(glm_rmse, 5), "\n")
cat("3. Ranger:  ", round(rf_rmse, 5), "\n")
cat("------------------------------------------------\n")


## --- 6. Blending and Submission ---

# Logaritmic predictions
pred_rf_log <- predict(rf_model_ranger, test_final_selected)
pred_glm_log <- predict(glm_model_final, test_final_selected)

# XGBoost wants matrix
test_matrix_sel <- as.matrix(test_final_selected)
pred_xgb_log <- predict(xgb_model, test_matrix_sel) 

# Weigthed blending (XGBoost 50%, Ranger 30%, GLMNet 20%)
pred_weighted_log <- (
  0.40 * pred_xgb_log + 
    0.25 * pred_rf_log + 
    0.35 * pred_glm_log
)

# original scale of prices
pred_final <- exp(pred_weighted_log)

# file final submission 
submission_blended <- data.frame(Id = test_ID, SalePrice = pred_final)
write.csv(submission_blended, "submission_blended.csv", row.names = FALSE)

cat("\n File 'submission_blended.csv' created using weighted blending (40/25/35).\n")


head(submission_blended)


