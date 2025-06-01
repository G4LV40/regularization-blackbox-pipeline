#---------------------------------------------------
# Bibliotecas
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
# Inicialização do H2O
#---------------------------------------------------
h2o.init(nthreads = -1)

#---------------------------------------------------
# Gerador de dados (Friedman 1991)
#---------------------------------------------------
gerar_dados <- function(n, p, seed = 42) {
  set.seed(seed)
  X <- matrix(runif(n * p), nrow = n)
  y <- 10 * sin(pi * X[,1] * X[,2]) + 
    20 * (X[,3] - 0.5)^2 +
    10 * X[,4] + 5 * X[,5] + rnorm(n)
  list(X = X, y = y)
}

#---------------------------------------------------
# Treinamento dos modelos
#---------------------------------------------------
avaliar_modelos_custom <- function(X, y, seed = 123) {
  set.seed(seed)
  n <- nrow(X)
  idx <- sample(1:n, size = 0.8 * n)
  X_train <- X[idx, ]; y_train <- y[idx]
  X_test  <- X[-idx, ]; y_test  <- y[-idx]
  
  resultados <- list()
  
  #----------------------------
  # Modelos Regularizados
  #----------------------------
  modelos_reg <- list(
    Ridge = cv.glmnet(X_train, y_train, alpha = 0),
    Lasso = cv.glmnet(X_train, y_train, alpha = 1),
    ElasticNet = cv.glmnet(X_train, y_train, alpha = 0.5)
  )
  
  for (nome in names(modelos_reg)) {
    modelo <- modelos_reg[[nome]]
    y_pred <- predict(modelo, X_test, s = "lambda.min")
    resultados[[nome]] <- rmse(y_test, y_pred)
    
    # Seleção de variáveis
    vars <- which(as.vector(coef(modelo, s = "lambda.min"))[-1] != 0)
    if (length(vars) < 2) next  # evita instabilidade
    
    Xtr_sel <- X_train[, vars, drop = FALSE]
    Xte_sel <- X_test[, vars, drop = FALSE]
    
    # Modelos híbridos: Black-box com variáveis selecionadas
    resultados[[paste0("RF_", nome)]] <- rmse(y_test, predict(randomForest(Xtr_sel, y_train), Xte_sel))
    
    dtrain <- xgb.DMatrix(data = Xtr_sel, label = y_train)
    dtest  <- xgb.DMatrix(data = Xte_sel)
    xgb <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror", verbose = 0)
    resultados[[paste0("XGBoost_", nome)]] <- rmse(y_test, predict(xgb, dtest))
    
    lgb_data <- lgb.Dataset(data = Xtr_sel, label = y_train)
    lgb <- lightgbm(data = lgb_data, nrounds = 100, objective = "regression", verbose = -1)
    resultados[[paste0("LightGBM_", nome)]] <- rmse(y_test, predict(lgb, Xte_sel))
    
    pool_train <- catboost.load_pool(Xtr_sel, label = y_train)
    pool_test  <- catboost.load_pool(Xte_sel)
    cb <- catboost.train(pool_train, NULL, params = list(loss_function = "RMSE", iterations = 100, verbose = 0))
    resultados[[paste0("CatBoost_", nome)]] <- rmse(y_test, catboost.predict(cb, pool_test))
    
    htrain <- as.h2o(as.data.frame(Xtr_sel) %>% mutate(y = y_train))
    htest  <- as.h2o(as.data.frame(Xte_sel))
    gbm <- h2o.gbm(y = "y", training_frame = htrain, ntrees = 100)
    yhat <- as.vector(h2o.predict(gbm, htest))
    resultados[[paste0("H2OGBM_", nome)]] <- rmse(y_test, yhat)
  }
  
  #----------------------------
  # Modelos Black-box puros (com todas as variáveis)
  #----------------------------
  resultados[["RF_Full"]] <- rmse(y_test, predict(randomForest(X_train, y_train), X_test))
  
  dtrain_full <- xgb.DMatrix(data = X_train, label = y_train)
  dtest_full  <- xgb.DMatrix(data = X_test)
  xgb_full <- xgboost(data = dtrain_full, nrounds = 100, objective = "reg:squarederror", verbose = 0)
  resultados[["XGBoost_Full"]] <- rmse(y_test, predict(xgb_full, dtest_full))
  
  lgb_data_full <- lgb.Dataset(data = X_train, label = y_train)
  lgb_full <- lightgbm(data = lgb_data_full, nrounds = 100, objective = "regression", verbose = -1)
  resultados[["LightGBM_Full"]] <- rmse(y_test, predict(lgb_full, X_test))
  
  pool_train_full <- catboost.load_pool(X_train, label = y_train)
  pool_test_full  <- catboost.load_pool(X_test)
  cb_full <- catboost.train(pool_train_full, NULL, params = list(loss_function = "RMSE", iterations = 100, verbose = 0))
  resultados[["CatBoost_Full"]] <- rmse(y_test, catboost.predict(cb_full, pool_test_full))
  
  htrain_full <- as.h2o(as.data.frame(X_train) %>% mutate(y = y_train))
  htest_full  <- as.h2o(as.data.frame(X_test))
  gbm_full <- h2o.gbm(y = "y", training_frame = htrain_full, ntrees = 100)
  resultados[["H2OGBM_Full"]] <- rmse(y_test, as.vector(h2o.predict(gbm_full, htest_full)))
  
  return(resultados)
}


#---------------------------------------------------
# Executar múltiplas simulações
#---------------------------------------------------

simular_multi <- function(n_sim = 10, n_vals = c(200, 500, 1000), p_vals = c(5, 10, 50)) {
  resultados <- list()
  for (n in n_vals) {
    for (p in p_vals) {
      for (sim in 1:n_sim) {
        dados <- gerar_dados(n, p, seed = 1000 + sim + n + p)
        res <- avaliar_modelos_custom(dados$X, dados$y)
        res_df <- tibble(
          sim = sim,
          n = n,
          p = p,
          modelo = names(res),
          rmse = unlist(res)
        )
        resultados[[length(resultados) + 1]] <- res_df
      }
    }
  }
  bind_rows(resultados)
}

#-----------------------------------------------
# Rodar simulações para todas combinações
#-----------------------------------------------
set.seed(123)
df_resultados <- simular_multi(n_sim = 10)

#save(df_resultados,file="resultados_simula.RDA")
write.csv(df_resultados, "resultados_simula.csv", row.names = FALSE)
#-----------------------------------------------
# Gráfico semelhante ao da imagem enviada
#-----------------------------------------------


# Lista de tamanhos de amostra únicos no seu dataset
valores_n <- unique(df_resultados$n)

# Loop para gerar uma imagem por valor de n
for (n_atual in valores_n) {
  
  # Filtrar os dados para o n atual
  dados_n <- subset(df_resultados, n == n_atual)
  
  # Criar o gráfico para esse valor de n
  p <- ggplot(dados_n, aes(x = factor(p), y = rmse, fill = modelo)) +
    geom_boxplot(outlier.size = 0.8, outlier.alpha = 0.4,
                 position = position_dodge2(preserve = "single")) +
    theme_minimal(base_size = 12) +
    labs(
      title = paste("RMSE Distribution by Model - n =", n_atual),
      x = "Total Number of variables (p)",
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
  
  # Exibir
  print(p)
  
  # Salvar imagem (opcional, descomente se quiser)
  # ggsave(paste0("rmse_modelos_n", n_atual, ".png"), plot = p, width = 10, height = 6, dpi = 300)
}



# Lista de tamanhos de amostra únicos no seu dataset
valores_n <- unique(df_resultados$n)

# Loop para gerar uma imagem por valor de n
for (n_atual in valores_n) {
  
  # Filtrar os dados para o n atual
  dados_n <- subset(df_resultados, n == n_atual)
  
  # Criar o gráfico com facet_grid por modelo
  p <- ggplot(dados_n, aes(x = factor(p), y = rmse, fill = modelo)) +
    geom_boxplot(outlier.size = 0.8, outlier.alpha = 0.4) +
    facet_wrap(~ modelo, nrow = 3) +  # ou use facet_grid(~ modelo) se quiser uma linha só
    theme_minimal(base_size = 12) +
    labs(
      title = paste("RMSE Distribution by Model - n =", n_atual),
      x = " ",
      y = "RMSE"
    ) +
    theme(
      strip.text = element_text(face = "bold", size = 11),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      legend.position = "none"  # oculta a legenda pois cada faceta já representa um modelo
    ) +
    scale_fill_viridis_d(option = "turbo")
  
  # Exibir
  print(p)
  
  # Salvar imagem (opcional, descomente se quiser)
  # ggsave(paste0("rmse_facet_n", n_atual, ".png"), plot = p, width = 12, height = 8, dpi = 300)
}







#simulacao_rmse_dotchart

# Dados
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

# Ordenar pelo RMSE
df <- df %>%
  arrange(RMSE) %>%
  mutate(Model = factor(Model, levels = Model))

# Dot chart horizontal
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

