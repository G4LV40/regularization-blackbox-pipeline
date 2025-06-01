
random_search <- function(model_type, X_train, Y_train, k) {
  search_grid <- list()
  
  # Definindo o grid de parâmetros para cada modelo
  if (model_type == "Random Forest") {
    search_grid <- expand.grid(
      ntree = sample(seq(100, 500, by = 100), 5),
      mtry = sample(seq(2, min(ncol(X_train), 10), by = 2), 5)
    )
    
  } else if (model_type == "XGBoost") {
    search_grid <- expand.grid(
      max_depth = sample(3:10, 5, replace = TRUE),
      eta = runif(5, 0.01, 0.3),
      nrounds = sample(seq(50, 500, by = 50), 5, replace = TRUE)
    )
    
  } else if (model_type == "H2O GBM") {
    search_grid <- expand.grid(
      ntrees = sample(seq(50, 500, by = 50), 5, replace = TRUE),
      max_depth = sample(3:10, 5, replace = TRUE)
    )
    
  } else if (model_type == "LightGBM") {
    search_grid <- expand.grid(
      num_leaves = sample(seq(10, 80, by = 10), 5, replace = TRUE),
      learning_rate = runif(5, 0.01, 0.3)
    )
    
  } else if (model_type == "CatBoost") {
    # Garantir mesmo comprimento nas colunas
    depths <- sample(4:10, 5, replace = TRUE)
    iterations <- sample(seq(50, 500, by = 50), 5, replace = TRUE)
    learning_rates <- runif(5, 0.01, 0.3)
    
    search_grid <- data.frame(
      depth = depths,
      iterations = iterations,
      learning_rate = learning_rates
    )
  }
  
  best_model <- NULL
  best_auc <- -Inf
  
  # Loop para avaliar todas as combinações de parâmetros
  for (i in 1:nrow(search_grid)) {
    params <- as.list(search_grid[i, ])
    
    # Avaliação do modelo usando K-Fold Cross Validation
    result <- train_and_evaluate_kfold(X_train, Y_train, model_type, params, k)
    
    # Se a AUC for melhor, salvar os melhores parâmetros
    if (result$auc > best_auc) {
      best_auc <- result$auc
      best_model <- params
    }
  }
  
  return(best_model)
}

