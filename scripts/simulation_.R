#---------------------------------------------------
# Libraries
#---------------------------------------------------
library(glmnet)
library(randomForest)
library(xgboost)
library(lightgbm)
library(catboost)
library(h2o)
library(dplyr)
library(tibble)
library(purrr)
library(Metrics)
library(tidyr)
library(ggplot2)
library(viridis)

#---------------------------------------------------
# H2O initialization
#---------------------------------------------------
h2o.init(nthreads = -1)

#---------------------------------------------------
# Data generator (Friedman 1991)
#---------------------------------------------------
generate_data <- function(n, p, seed = 42) {
  set.seed(seed)
  X <- matrix(runif(n * p), nrow = n)
  y <- 10 * sin(pi * X[,1] * X[,2]) + 
    20 * (X[,3] - 0.5)^2 +
    10 * X[,4] + 5 * X[,5] + rnorm(n)
  list(X = X, y = y)
}

#---------------------------------------------------
# Model training and evaluation
#---------------------------------------------------
evaluate_custom_models <- function(X, y, seed = 123) {
  set.seed(seed)
  n <- nrow(X)
  idx <- sample(1:n, size = 0.8 * n)
  X_train <- X[idx, ]; y_train <- y[idx]
  X_test  <- X[-idx, ]; y_test  <- y[-idx]
  
  results <- list()
  
  #----------------------------
  # Regularized models
  #----------------------------
  reg_models <- list(
    Ridge = cv.glmnet(X_train, y_train, alpha = 0),
    Lasso = cv.glmnet(X_train, y_train, alpha = 1),
    ElasticNet = cv.glmnet(X_train, y_train, alpha = 0.5)
  )
  
  for (name in names(reg_models)) {
    model <- reg_models[[name]]
    y_pred <- predict(model, X_test, s = "lambda.min")
    results[[name]] <- rmse(y_test, y_pred)
    
    # Variable selection
    vars <- which(as.vector(coef(model, s = "lambda.min"))[-1] != 0)
    if (length(vars) < 2) next  # avoids instability
    
    Xtr_sel <- X_train[, vars, drop = FALSE]
    Xte_sel <- X_test[, vars, drop = FALSE]
    
    # Hybrid models: black-box models using selected variables
    results[[paste0("RF_", name)]] <- rmse(y_test, predict(randomForest(Xtr_sel, y_train), Xte_sel))
    
    dtrain <- xgb.DMatrix(data = Xtr_sel, label = y_train)
    dtest  <- xgb.DMatrix(data = Xte_sel)
    xgb <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror", verbose = 0)
    results[[paste0("XGBoost_", name)]] <- rmse(y_test, predict(xgb, dtest))
    
    lgb_data <- lgb.Dataset(data = Xtr_sel, label = y_train)
    lgb <- lightgbm(data = lgb_data, nrounds = 100, objective = "regression", verbose = -1)
    results[[paste0("LightGBM_", name)]] <- rmse(y_test, predict(lgb, Xte_sel))
    
    pool_train <- catboost.load_pool(Xtr_sel, label = y_train)
    pool_test  <- catboost.load_pool(Xte_sel)
    cb <- catboost.train(pool_train, NULL, params = list(loss_function = "RMSE", iterations = 100, verbose = 0))
    results[[paste0("CatBoost_", name)]] <- rmse(y_test, catboost.predict(cb, pool_test))
    
    htrain <- as.h2o(as.data.frame(Xtr_sel) %>% mutate(y = y_train))
    htest  <- as.h2o(as.data.frame(Xte_sel))
    gbm <- h2o.gbm(y = "y", training_frame = htrain, ntrees = 100)
    yhat <- as.vector(h2o.predict(gbm, htest))
    results[[paste0("H2OGBM_", name)]] <- rmse(y_test, yhat)
  }
  
  #----------------------------
  # Full black-box models (using all features)
  #----------------------------
  results[["RF_Full"]] <- rmse(y_test, predict(randomForest(X_train, y_train), X_test))
  
  dtrain_full <- xgb.DMatrix(data = X_train, label = y_train)
  dtest_full  <- xgb.DMatrix(data = X_test)
  xgb_full <- xgboost(data = dtrain_full, nrounds = 100, objective = "reg:squarederror", verbose = 0)
  results[["XGBoost_Full"]] <- rmse(y_test, predict(xgb_full, dtest_full))
  
  lgb_data_full <- lgb.Dataset(data = X_train, label = y_train)
  lgb_full <- lightgbm(data = lgb_data_full, nrounds = 100, objective = "regression", verbose = -1)
  results[["LightGBM_Full"]] <- rmse(y_test, predict(lgb_full, X_test))
  
  pool_train_full <- catboost.load_pool(X_train, label = y_train)
  pool_test_full  <- catboost.load_pool(X_test)
  cb_full <- catboost.train(pool_train_full, NULL, params = list(loss_function = "RMSE", iterations = 100, verbose = 0))
  results[["CatBoost_Full"]] <- rmse(y_test, catboost.predict(cb_full, pool_test_full))
  
  htrain_full <- as.h2o(as.data.frame(X_train) %>% mutate(y = y_train))
  htest_full  <- as.h2o(as.data.frame(X_test))
  gbm_full <- h2o.gbm(y = "y", training_frame = htrain_full, ntrees = 100)
  results[["H2OGBM_Full"]] <- rmse(y_test, as.vector(h2o.predict(gbm_full, htest_full)))
  
  return(results)
}

#---------------------------------------------------
# Run multiple simulations
#---------------------------------------------------
simulate_multiple <- function(n_sim = 10, n_vals = c(200, 500, 1000), p_vals = c(5, 10, 50)) {
  results <- list()
  for (n in n_vals) {
    for (p in p_vals) {
      for (sim in 1:n_sim) {
        data <- generate_data(n, p, seed = 1000 + sim + n + p)
        res <- evaluate_custom_models(data$X, data$y)
        res_df <- tibble(
          sim = sim,
          n = n,
          p = p,
          model = names(res),
          rmse = unlist(res)
        )
        results[[length(results) + 1]] <- res_df
      }
    }
  }
  bind_rows(results)
}

#---------------------------------------------------
# Run simulations for all combinations
#---------------------------------------------------
set.seed(123)
df_results <- simulate_multiple(n_sim = 10)

# Export results
# save(df_results, file = "results_simulation.RDA")
write.csv(df_results, "results_simulation.csv", row.names = FALSE)

#---------------------------------------------------
# Boxplot by n (panel 1)
#---------------------------------------------------
sample_sizes <- unique(df_results$n)

for (current_n in sample_sizes) {
  data_n <- subset(df_results, n == current_n)
  
  p <- ggplot(data_n, aes(x = factor(p), y = rmse, fill = model)) +
    geom_boxplot(outlier.size = 0.8, outlier.alpha = 0.4,
                 position = position_dodge2(preserve = "single")) +
    theme_minimal(base_size = 12) +
    labs(
      title = paste("RMSE Distribution by Model - n =", current_n),
      x = "Total Number of Variables (p)",
      y = "RMSE",
      fill = "Model"
    ) +
    theme(
      strip.text = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      legend.position = "bottom",
      legend.box = "vertical"
    ) +
    scale_fill_viridis_d(option = "turbo")
  
  print(p)
  # ggsave(paste0("rmse_models_n", current_n, ".png"), plot = p, width = 10, height = 6, dpi = 300)
}

#---------------------------------------------------
# Boxplot using facet_wrap by model (panel 2)
#---------------------------------------------------
for (current_n in sample_sizes) {
  data_n <- subset(df_results, n == current_n)
  
  p <- ggplot(data_n, aes(x = factor(p), y = rmse, fill = model)) +
    geom_boxplot(outlier.size = 0.8, outlier.alpha = 0.4) +
    facet_wrap(~ model, nrow = 3) +
    theme_minimal(base_size = 12) +
    labs(
      title = paste("RMSE Distribution by Model - n =", current_n),
      x = " ",
      y = "RMSE"
    ) +
    theme(
      strip.text = element_text(face = "bold", size = 11),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      legend.position = "none"
    ) +
    scale_fill_viridis_d(option = "turbo")
  
  print(p)
  # ggsave(paste0("rmse_facet_n", current_n, ".png"), plot = p, width = 12, height = 8, dpi = 300)
}

#---------------------------------------------------
# Dot chart: mean RMSE by model
#---------------------------------------------------
df <- data.frame(
  Model = c(
    "CatBoost_Full", "CatBoost_Ridge", "LightGBM_Full", "LightGBM_Ridge",
    "H2OGBM_Ridge", "H2OGBM_Full", "CatBoost_ElasticNet", "LightGBM_ElasticNet",
    "CatBoost_Lasso", "H2OGBM_ElasticNet", "LightGBM_Lasso", "H2OGBM_Lasso",
    "XGBoost_Full", "XGBoost_Ridge", "XGBoost_ElasticNet", "XGBoost_Lasso",
    "RF_ElasticNet", "RF_Lasso", "RF_Full", "RF_Ridge",
    "Ridge", "Lasso", "ElasticNet"
  ),
  RMSE = c(
    1.7506, 1.7506, 1.7654, 1.7654, 1.8453, 1.8453, 1.8600, 1.9058,
    1.9134, 1.9598, 1.9715, 2.0141, 2.1672, 2.1672, 2.2206, 2.2884,
    2.4479, 2.4631, 2.4661, 2.4661, 2.6771, 2.7190, 2.7527
  )
)

df <- df %>%
  arrange(RMSE) %>%
  mutate(Model = factor(Model, levels = Model))

ggplot(df, aes(x = RMSE, y = Model)) +
  geom_segment(aes(x = min(RMSE), xend = RMSE, y = Model, yend = Model),
               color = "gray80") +
  geom_point(size = 2.5, color = "black") +
  labs(
    title = "Model Performance (Sorted by RMSE)",
    x = "Mean RMSE",
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.major.y = element_blank())
