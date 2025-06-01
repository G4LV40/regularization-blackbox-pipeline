#---------------------------------------------------
# Function to train and evaluate a classification model
#---------------------------------------------------
train_and_evaluate <- function(X_train, Y_train, X_test, Y_test, model_type, params) {
  Y_train <- factor(Y_train, levels = c("N", "S"))
  Y_test  <- factor(Y_test,  levels = c("N", "S"))
  
  if (model_type == "Random Forest") {
    # Set hyperparameters or use defaults
    ntree_val <- if (!is.null(params$ntree)) params$ntree else 100
    mtry_val  <- if (!is.null(params$mtry) && params$mtry >= 1 && params$mtry <= ncol(X_train)) {
      params$mtry
    } else {
      floor(sqrt(ncol(X_train)))
    }
    # Train model
    model <- randomForest(x = X_train, y = Y_train, ntree = ntree_val, mtry = mtry_val)
    prob_predictions <- predict(model, X_test, type = "prob")
    predictions <- predict(model, X_test)
    
  } else if (model_type == "XGBoost") {
    # Prepare data for XGBoost
    dtrain <- xgb.DMatrix(as.matrix(X_train), label = as.numeric(Y_train) - 1)
    dtest  <- xgb.DMatrix(as.matrix(X_test))
    max_depth_val <- ifelse(!is.null(params$max_depth), params$max_depth, 6)
    eta_val       <- ifelse(!is.null(params$eta), params$eta, 0.1)
    nrounds_val   <- ifelse(!is.null(params$nrounds), params$nrounds, 100)
    
    # Train model
    model <- xgb.train(
      params = list(objective = "binary:logistic", eval_metric = "auc", 
                    max_depth = max_depth_val, eta = eta_val),
      data = dtrain, nrounds = nrounds_val
    )
    prob <- predict(model, dtest)
    predictions <- factor(ifelse(prob > 0.5, "S", "N"), levels = c("N", "S"))
    prob_predictions <- data.frame(N = 1 - prob, S = prob)
    
  } else if (model_type == "H2O GBM") {
    # Prepare data for H2O
    train_h2o <- as.h2o(data.frame(X_train, S = Y_train))
    test_h2o  <- as.h2o(data.frame(X_test, S = Y_test))
    ntrees_val     <- ifelse(!is.null(params$ntrees), params$ntrees, 100)
    max_depth_val  <- ifelse(!is.null(params$max_depth), params$max_depth, 6)
    
    # Train model
    model <- h2o.gbm(x = colnames(X_train), y = "S", training_frame = train_h2o,
                     ntrees = ntrees_val, max_depth = max_depth_val)
    
    # Predict
    pred_h2o <- as.data.frame(h2o.predict(model, test_h2o))
    predictions <- factor(pred_h2o$predict, levels = c("N", "S"))
    prob_predictions <- data.frame(N = 1 - pred_h2o$S, S = pred_h2o$S)
    
  } else if (model_type == "LightGBM") {
    # Prepare data for LightGBM
    train_data <- lgb.Dataset(as.matrix(X_train), label = as.numeric(Y_train) - 1)
    num_leaves_val   <- ifelse(!is.null(params$num_leaves), params$num_leaves, 31)
    learning_rate_val <- ifelse(!is.null(params$learning_rate), params$learning_rate, 0.1)
    nrounds_val       <- ifelse(!is.null(params$nrounds), params$nrounds, 100)
    
    # Train model
    model <- lgb.train(
      params = list(objective = "binary", metric = "auc",
                    num_leaves = num_leaves_val, learning_rate = learning_rate_val),
      data = train_data, nrounds = nrounds_val
    )
    
    prob <- predict(model, as.matrix(X_test))
    predictions <- factor(ifelse(prob > 0.5, "S", "N"), levels = c("N", "S"))
    prob_predictions <- data.frame(N = 1 - prob, S = prob)
    
  } else if (model_type == "CatBoost") {
    # Prepare data for CatBoost
    train_pool <- catboost.load_pool(data = as.matrix(X_train), label = as.numeric(Y_train) - 1)
    test_pool  <- catboost.load_pool(data = as.matrix(X_test))
    
    iterations_val     <- ifelse(!is.null(params$iterations), params$iterations, 100)
    depth_val          <- ifelse(!is.null(params$depth), params$depth, 6)
    learning_rate_val  <- ifelse(!is.null(params$learning_rate), params$learning_rate, 0.1)
    
    # Train model
    model <- catboost.train(train_pool, params = list(
      iterations = iterations_val,
      depth = depth_val,
      learning_rate = learning_rate_val,
      loss_function = 'Logloss',
      custom_metric = list('AUC')
    ))
    
    prob <- catboost.predict(model, test_pool, prediction_type = "Probability")
    predictions <- factor(ifelse(prob > 0.5, "S", "N"), levels = c("N", "S"))
    prob_predictions <- data.frame(N = 1 - prob, S = prob)
  }
  
  # Return evaluation metrics
  return(evaluate_model(predictions, prob_predictions, Y_test))
}


#---------------------------------------------------
# Function to perform k-Fold Cross-Validation
#---------------------------------------------------
train_and_evaluate_kfold <- function(X, Y, model_type, params, k ) {
  folds <- createFolds(Y, k = k)
  
  # Apply training and evaluation for each fold
  metrics <- lapply(folds, function(train_idx) {
    X_train <- X[train_idx, , drop = FALSE]
    Y_train <- Y[train_idx]
    X_test <- X[-train_idx, , drop = FALSE]
    Y_test <- Y[-train_idx]
    
    train_and_evaluate(X_train, Y_train, X_test, Y_test, model_type, params)
  })
  
  # Compute mean of each metric
  mean_metrics <- function(metric) mean(sapply(metrics, `[[`, metric), na.rm = TRUE)
  
  return(list(
    precision = mean_metrics("precision"),
    recall = mean_metrics("recall"),
    f1_score = mean_metrics("f1_score"),
    auc = mean_metrics("auc")
  ))
}
