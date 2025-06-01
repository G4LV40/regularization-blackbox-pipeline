#---------------------------------------------------
# Bibliotecas necessárias
#-------------------------------------------------------------------------------
library(caret)
library(pROC)
library(randomForest)
library(xgboost)
library(h2o)
library(lightgbm)
library(catboost)

#-------------------------------------------------------------------------------
# Configurações iniciais
#-------------------------------------------------------------------------------
set.seed(42)      # Definir semente para reprodutibilidade
h2o.init()        # Inicializar o cluster H2O

#-------------------------------------------------------------------------------
# Carregamento dos dados
#-------------------------------------------------------------------------------
load(file = "TravelInsurancePrediction.RData")  # Carregar dataset
dados <- df
rm(df)             # Remover objeto original para liberar memória
gc()               # Limpeza de memória (garbage collection)

#banco de amostragem
#source("amostra_100.R")


#-------------------------------------------------------------------------------
# Divisão dos dados em treino e teste
#---------------------------------------------------
inTraining <- createDataPartition(dados$TravelInsurance, p = 0.80, list = FALSE)
training <- dados[inTraining, ]
testing <- dados[-inTraining, ]

# Verificar a distribuição da variável resposta
table(training$TravelInsurance)
table(testing$TravelInsurance)

# Definicao do numero de folds
k= 5

#-------------------------------------------------------------------------------
# Carregar variáveis selecionadas via regularização
#-------------------------------------------------------------------------------
load("Ridge_model_evaluation.RData")
ridge_features <- rownames(varImp(model_Ridge, scale = FALSE)$importance)

load("Lasso_model_evaluation.RData")
lasso_features <- rownames(varImp(model_Lasso, scale = FALSE)$importance)

load("ElasticNet_model_evaluation.RData")
elasticnet_features <- rownames(varImp(model_EN, scale = FALSE)$importance)

# Agrupar os conjuntos de variáveis
feature_sets <- list(Ridge = ridge_features,
                     Lasso = lasso_features,
                     ElasticNet = elasticnet_features)

#-------------------------------------------------------------------------------
# Definição dos modelos a serem avaliados-
#---------------------------------------------------
model_types <- c("Random Forest", "XGBoost", "H2O GBM", "LightGBM", "CatBoost")

#-------------------------------------------------------------------------------
# Funções auxiliares
#-------------------------------------------------------------------------------

#### Função para calcular métricas de avaliação do modelo

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


# Função para instanciar os modelos comparativos
source("funct_Models_.R")

# Função para otimização de hiperparâmetros via Random Search
source("funct_RND_search.R")



#---------------------------------------------------
# Execução principal: Treinamento e Avaliação Final 
#---------------------------------------------------

best_results <- NULL  # Inicializar resultados

# Loop para otimizar cada modelo com cada conjunto de features
for (model_type in model_types) {
  for (regularization in names(feature_sets)) {
    
    features <- feature_sets[[regularization]]
    if (!all(features %in% colnames(training))) next
    
    X_train <- training[, features, drop = FALSE]
    Y_train <- training$TravelInsurance
    
    cat("Otimizando modelo:", model_type,
        "com features de:", regularization, "\n")
    
    # 1. Busca hiperparâmetros via Random Search + K-Fold
    best_params <- random_search(model_type, X_train, Y_train, k)
    
    # 2. Avaliação final também via K-Fold
    result <- train_and_evaluate_kfold(X_train,
                                       Y_train,
                                       model_type,
                                       best_params,
                                       k )
    
    # 3. Monta linha de resultados
    result_row <- data.frame(
      Model = model_type,
      Regularization = regularization,
      AUC = result$auc,
      Precision = result$precision,
      Recall = result$recall,
      F1_Score = result$f1_score
    )
    
    if (!is.null(best_params)) {
      params_df <- as.data.frame(t(best_params))
      result_row <- cbind(result_row, params_df)
    }
    
    # 4. Armazena resultados de forma segura
    if (is.null(best_results)) {
      best_results <- result_row
    } else {
      common_cols <- union(names(best_results), names(result_row))
      best_results <- merge(best_results, result_row, all = TRUE)
    }
  }
}

#---------------------------------------------------
# Visualização dos Resultados Finais (médias dos folds)
#---------------------------------------------------
print(best_results)

