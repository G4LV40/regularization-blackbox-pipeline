#---------------------------------------------------
# Bibliotecas necessárias
#---------------------------------------------------
library(caret)
library(pROC)
library(randomForest)
library(xgboost)
library(h2o)
library(lightgbm)
library(catboost)
library(dplyr)
library(ggplot2)
library(tidyr)
library(knitr)

#---------------------------------------------------
# Configurações iniciais
#---------------------------------------------------
set.seed(42)
h2o.init()

#---------------------------------------------------
# Carregamento dos dados
#---------------------------------------------------
load(file = "TravelInsurancePrediction.RData")
dados <- df
rm(df)
gc()

#---------------------------------------------------
# Divisão dos dados em treino e teste
#---------------------------------------------------
inTraining <- createDataPartition(dados$TravelInsurance, p = 0.80, list = FALSE)
training <- dados[inTraining, ]
testing <- dados[-inTraining, ]

# Definição do número de folds
k <- 5

#---------------------------------------------------
# Modelos Black-Box
#---------------------------------------------------
model_types <- c("Random Forest", "XGBoost", "H2O GBM", "LightGBM", "CatBoost")

#---------------------------------------------------
# Funções auxiliares
#---------------------------------------------------

# Avaliação do modelo
evaluate_model <- function(predictions, prob_predictions, true_labels) {
  roc_obj <- roc(response = true_labels, predictor = prob_predictions[, "S"])
  auc_value <- auc(roc_obj)
  
  conf_matrix <- confusionMatrix(predictions, true_labels)
  precision <- conf_matrix$byClass['Pos Pred Value']
  recall <- conf_matrix$byClass['Sensitivity']
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  return(list(precision = precision, recall = recall, f1_score = f1_score, auc = auc_value))
}

# Treinamento e avaliação de um modelo
train_and_evaluate <- function(X_train, Y_train, X_test, Y_test, model_type, params) {
  Y_train <- factor(Y_train, levels = c("N", "S"))
  Y_test <- factor(Y_test, levels = c("N", "S"))
  
  if (model_type == "Random Forest") {
    model <- randomForest(x = X_train, y = Y_train, ntree = params$ntree, mtry = params$mtry)
    prob_predictions <- predict(model, X_test, type = "prob")
    predictions <- predict(model, X_test)
    
  } else if (model_type == "XGBoost") {
    dtrain <- xgb.DMatrix(as.matrix(X_train), label = as.numeric(Y_train) - 1)
    dtest <- xgb.DMatrix(as.matrix(X_test))
    
    model <- xgb.train(params = list(objective = "binary:logistic", eval_metric = "auc", 
                                     max_depth = params$max_depth, eta = params$eta), 
                       data = dtrain, nrounds = params$nrounds)
    
    prob <- predict(model, dtest)
    predictions <- factor(ifelse(prob > 0.5, "S", "N"), levels = c("N", "S"))
    prob_predictions <- data.frame(N = 1 - prob, S = prob)
    
  } else if (model_type == "H2O GBM") {
    train_h2o <- as.h2o(data.frame(X_train, S = Y_train))
    test_h2o <- as.h2o(data.frame(X_test, S = Y_test))
    
    model <- h2o.gbm(x = colnames(X_train), y = "S", training_frame = train_h2o,
                     ntrees = params$ntrees, max_depth = params$max_depth)
    
    pred_h2o <- as.data.frame(h2o.predict(model, test_h2o))
    predictions <- factor(pred_h2o$predict, levels = c("N", "S"))
    prob_predictions <- data.frame(N = 1 - pred_h2o$S, S = pred_h2o$S)
    
  } else if (model_type == "LightGBM") {
    train_data <- lgb.Dataset(as.matrix(X_train), label = as.numeric(Y_train) - 1)
    
    model <- lgb.train(params = list(objective = "binary", metric = "auc",
                                     num_leaves = params$num_leaves, learning_rate = params$learning_rate),
                       data = train_data, nrounds = params$nrounds)
    
    prob <- predict(model, as.matrix(X_test))
    predictions <- factor(ifelse(prob > 0.5, "S", "N"), levels = c("N", "S"))
    prob_predictions <- data.frame(N = 1 - prob, S = prob)
    
  } else if (model_type == "CatBoost") {
    train_pool <- catboost.load_pool(data = as.matrix(X_train), label = as.numeric(Y_train) - 1)
    test_pool <- catboost.load_pool(data = as.matrix(X_test))
    
    model <- catboost.train(train_pool, params = list(iterations = params$iterations,
                                                      depth = params$depth,
                                                      learning_rate = params$learning_rate,
                                                      loss_function = 'Logloss',
                                                      custom_metric = list('AUC')))
    
    prob <- catboost.predict(model, test_pool, prediction_type = "Probability")
    predictions <- factor(ifelse(prob > 0.5, "S", "N"), levels = c("N", "S"))
    prob_predictions <- data.frame(N = 1 - prob, S = prob)
  }
  
  return(evaluate_model(predictions, prob_predictions, Y_test))
}

# Avaliação k-Fold
train_and_evaluate_kfold <- function(X, Y, model_type, params, k) {
  folds <- createFolds(Y, k = k)
  
  metrics <- lapply(folds, function(train_idx) {
    X_train <- X[train_idx, , drop = FALSE]
    Y_train <- Y[train_idx]
    X_test <- X[-train_idx, , drop = FALSE]
    Y_test <- Y[-train_idx]
    
    train_and_evaluate(X_train, Y_train, X_test, Y_test, model_type, params)
  })
  
  mean_metrics <- function(metric) mean(sapply(metrics, `[[`, metric), na.rm = TRUE)
  
  return(list(
    precision = mean_metrics("precision"),
    recall = mean_metrics("recall"),
    f1_score = mean_metrics("f1_score"),
    auc = mean_metrics("auc")
  ))
}

# Random Search
random_search <- function(model_type, X_train, Y_train, k) {
  grid <- switch(model_type,
                 "Random Forest" = expand.grid(ntree = sample(100:500, 5),
                                               mtry = sample(2:10, 5)),
                 "XGBoost" = expand.grid(max_depth = sample(3:10, 5),
                                         eta = runif(5, 0.01, 0.3),
                                         nrounds = sample(50:500, 5)),
                 "H2O GBM" = expand.grid(ntrees = sample(50:500, 5),
                                         max_depth = sample(3:10, 5)),
                 "LightGBM" = expand.grid(num_leaves = sample(10:80, 5),
                                          learning_rate = runif(5, 0.01, 0.3),
                                          nrounds = sample(50:500, 5)),
                 "CatBoost" = expand.grid(depth = sample(4:10, 5),
                                          iterations = sample(50:500, 5), 
                                          learning_rate = runif(5, 0.01, 0.3))
  )
  
  best_auc <- -Inf
  best_params <- NULL
  
  for (i in 1:nrow(grid)) {
    params <- as.list(grid[i, ])
    result <- train_and_evaluate_kfold(X_train, Y_train, model_type, params, k)
    
    if (result$auc > best_auc) {
      best_auc <- result$auc
      best_params <- params
    }
  }
  
  return(best_params)
}

# Seleção de Features por importância

get_top_features <- function(X, Y, model_type, top_n = 20) {
  Y <- factor(Y, levels = c("N", "S"))  # Corrige o problema do tipo
  
  if (model_type == "Random Forest") {
    model <- randomForest(x = X, y = Y, ntree = 200)
    importance_vals <- importance(model)[, 1]
    
  } else if (model_type == "XGBoost") {
    dtrain <- xgb.DMatrix(as.matrix(X), label = as.numeric(Y) - 1)
    model <- xgb.train(params = list(objective = "binary:logistic"), data = dtrain, nrounds = 200)
    imp <- xgb.importance(model = model)
    importance_vals <- imp$Gain
    names(importance_vals) <- imp$Feature
    
  } else if (model_type == "H2O GBM") {
    train_h2o <- as.h2o(data.frame(X, S = Y))
    model <- h2o.gbm(x = colnames(X), y = "S", training_frame = train_h2o, ntrees = 200)
    imp <- as.data.frame(h2o.varimp(model))
    importance_vals <- imp$relative_importance
    names(importance_vals) <- imp$variable
    
  } else if (model_type == "LightGBM") {
    train_data <- lgb.Dataset(as.matrix(X), label = as.numeric(Y) - 1)
    model <- lgb.train(params = list(objective = "binary", metric = "auc"), data = train_data, nrounds = 200)
    imp <- lgb.importance(model)
    importance_vals <- imp$Gain
    names(importance_vals) <- imp$Feature
    
  } else if (model_type == "CatBoost") {
    train_pool <- catboost.load_pool(data = as.matrix(X), label = as.numeric(Y) - 1)
    model <- catboost.train(train_pool, params = list(loss_function = 'Logloss', iterations = 200))
    imp <- catboost.get_feature_importance(model)
    importance_vals <- imp
    names(importance_vals) <- colnames(X)
    
  } else {
    stop("Modelo não reconhecido para seleção de variáveis:", model_type)
  }
  
  # Selecionar apenas as top variáveis
  top_features <- names(sort(importance_vals, decreasing = TRUE))[1:min(top_n, length(importance_vals))]
  return(top_features)
}


#---------------------------------------------------
# Execução principal com validação do número ótimo de features
#---------------------------------------------------
best_results <- data.frame()
auc_log_all <- list()

for (model_type in model_types) {
  auc_log <- data.frame(N = integer(), AUC = numeric())
  best_auc <- -Inf
  best_n <- NA
  best_result <- NULL
  best_params_final <- NULL
  
  for (n in seq(5, 50, 5)) {
    features <- get_top_features(training[, -which(names(training) == "TravelInsurance")], 
                                 training$TravelInsurance, model_type, top_n = n)
    X_train <- training[, features, drop = FALSE]
    Y_train <- training$TravelInsurance
    
    best_params <- random_search(model_type, X_train, Y_train, k)
    result <- train_and_evaluate_kfold(X_train, Y_train, model_type, best_params, k)
    
    auc_log <- rbind(auc_log, data.frame(N = n, AUC = result$auc))
    
    if (result$auc > best_auc) {
      best_auc <- result$auc
      best_n <- n
      best_result <- result
      best_params_final <- best_params
    }
  }
  
  final_features <- get_top_features(training[, -which(names(training) == "TravelInsurance")], 
                                     training$TravelInsurance, model_type, top_n = best_n)
  
  result_row <- data.frame(
    Model = model_type,
    Num_Features = best_n,
    AUC = best_result$auc,
    Precision = best_result$precision,
    Recall = best_result$recall,
    F1_Score = best_result$f1_score
  )
  
  if (!is.null(best_params_final)) {
    params_df <- as.data.frame(t(best_params_final))
    result_row <- cbind(result_row, params_df)
  }
  
  best_results <- rbind(best_results, result_row)
  auc_log_all[[model_type]] <- auc_log
}

#---------------------------------------------------
# Visualização final
#---------------------------------------------------
kable(best_results, caption = "Melhores modelos por técnica e número ótimo de variáveis")

# Gráficos de AUC vs. Número de Features por modelo
for (model in names(auc_log_all)) {
  auc_data <- auc_log_all[[model]]
  best_row <- auc_data[which.max(auc_data$AUC), ]
  print(
    ggplot(auc_data, aes(x = N, y = AUC)) +
      geom_line(color = "steelblue") +
      geom_point() +
      geom_vline(xintercept = best_row$N, linetype = "dashed", color = "red") +
      geom_point(aes(x = best_row$N, y = best_row$AUC), color = "red", size = 3) +
      annotate("text", x = best_row$N, y = best_row$AUC + 0.002,
               label = paste("Ótimo:", best_row$N, "variáveis"),
               color = "red", hjust = -0.1, size = 4.5) +
      labs(title = paste("AUC vs Número de Features -", model),
           x = "Número de Variáveis Selecionadas", y = "AUC") +
      theme_minimal(base_size = 14)
  )
}


