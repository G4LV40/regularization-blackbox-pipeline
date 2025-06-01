#---------------------------------------------------
# Required Libraries
#---------------------------------------------------
library(caret)
library(pROC)
library(randomForest)
library(xgboost)
library(h2o)
library(lightgbm)
library(catboost)

#---------------------------------------------------
# Initial Configuration
#---------------------------------------------------
set.seed(42)      # Set seed for reproducibility
h2o.init()        # Initialize H2O cluster

#---------------------------------------------------
# Data Loading
#---------------------------------------------------
load(file = "TravelInsurancePrediction.RData")  # Load dataset
df_1 <- df
rm(df)             # Remove original object to free memory
gc()               # Garbage collection


#---------------------------------------------------
# Split data into training and test sets
#---------------------------------------------------
inTraining <- createDataPartition(df_1$TravelInsurance, p = 0.80, list = FALSE)
training <- df_1[inTraining, ]
testing <- df_1[-inTraining, ]

# Check distribution of target variable
table(training$TravelInsurance)
table(testing$TravelInsurance)

# Define number of folds
k = 5

#---------------------------------------------------
# Load features selected via regularization
#---------------------------------------------------
load("Ridge_model_evaluation.RData")
ridge_features <- rownames(varImp(model_Ridge, scale = FALSE)$importance)

load("Lasso_model_evaluation.RData")
lasso_features <- rownames(varImp(model_Lasso, scale = FALSE)$importance)

load("ElasticNet_model_evaluation.RData")
elasticnet_features <- rownames(varImp(model_EN, scale = FALSE)$importance)

# Group feature sets
feature_sets <- list(Ridge = ridge_features,
                     Lasso = lasso_features,
                     ElasticNet = elasticnet_features)

#---------------------------------------------------
# Define model types to be evaluated
#---------------------------------------------------
model_types <- c("Random Forest", "XGBoost", "H2O GBM", "LightGBM", "CatBoost")

#---------------------------------------------------
# Auxiliary Functions
#---------------------------------------------------

# Function to compute evaluation metrics
evaluate_model <- function(predictions, prob_predictions, true_labels) {
  roc_obj <- roc(response = true_labels, predictor = prob_predictions[, "S"])
  auc_value <- auc(roc_obj)
  
  conf_matrix <- confusionMatrix(predictions, true_labels)
  precision <- conf_matrix$byClass['Pos Pred Value']
  recall <- conf_matrix$byClass['Sensitivity']
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  return(list(precision = precision,
              recall = recall, 
              f1_score = f1_score,
              auc = auc_value))
}

# Load model training functions
source("funct_Models_.R")

# Load random search function
source("funct_RND_search.R")

#---------------------------------------------------
# Main Execution: Training and Final Evaluation
#---------------------------------------------------

best_results <- NULL  # Initialize results object

# Loop through each model and feature set combination
for (model_type in model_types) {
  for (regularization in names(feature_sets)) {
    
    features <- feature_sets[[regularization]]
    if (!all(features %in% colnames(training))) next
    
    X_train <- training[, features, drop = FALSE]
    Y_train <- training$TravelInsurance
    
    cat("Optimizing model:", model_type,
        "with features from:", regularization, "\n")
    
    # 1. Hyperparameter tuning via Random Search + k-Fold
    best_params <- random_search(model_type, X_train, Y_train, k)
    
    # 2. Final evaluation also via k-Fold
    result <- train_and_evaluate_kfold(X_train,
                                       Y_train,
                                       model_type,
                                       best_params,
                                       k )
    
    # 3. Compile result row
    result_row <- data.frame(
      Model = model_type,
      Regularization = regularization,
      AUC = result$auc,
      Precision = result$precision,
      Recall = result$recall,
      F1_Score = result$f1_score
    )
    
    # Append hyperparameters to result row
    if (!is.null(best_params)) {
      params_df <- as.data.frame(t(best_params))
      result_row <- cbind(result_row, params_df)
    }
    
    # 4. Safely store the results
    if (is.null(best_results)) {
      best_results <- result_row
    } else {
      common_cols <- union(names(best_results), names(result_row))
      best_results <- merge(best_results, result_row, all = TRUE)
    }
  }
}

#---------------------------------------------------
# Visualization of Final Results (fold means)
#---------------------------------------------------
print(best_results)
