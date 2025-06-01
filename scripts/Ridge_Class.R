#---------------------------------------------------
# Bibliotecas necessárias
#---------------------------------------------------
library(caret)
library(pROC)
library(glmnet)
library(ggplot2)

#---------------------------------------------------
# Configurações iniciais
#---------------------------------------------------
set.seed(42)

#---------------------------------------------------
# Carregamento dos dados
#---------------------------------------------------
load(file = "TravelInsurancePrediction.RData")
dados <- df
rm(df)
gc()

#---------------------------------------------------
# Pré-processamento
#---------------------------------------------------
dados <- dados[, -20]  # Remover coluna não especificada
dados$TravelInsurance <- as.factor(dados$TravelInsurance)

#---------------------------------------------------
# Divisão treino/teste
#---------------------------------------------------
inTraining <- createDataPartition(dados$TravelInsurance, p = 0.80, list = FALSE)
training <- dados[inTraining, ]
testing <- dados[-inTraining, ]

#---------------------------------------------------
# Controle de treinamento
#---------------------------------------------------
ctrl <- trainControl(method = "cv",
                     number = 10,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     verboseIter = FALSE,
                     allowParallel = TRUE)

#---------------------------------------------------
# Treinamento do modelo Ridge com todas as variáveis
#---------------------------------------------------
X <- training[, -which(names(training) == "TravelInsurance")]
Y <- training$TravelInsurance

model_full <- train(x = X, y = Y,
                    method = "glmnet",
                    metric = "ROC",
                    trControl = ctrl)

#---------------------------------------------------
# Coeficientes do melhor modelo Ridge
#---------------------------------------------------
coef_Ridge <- as.matrix(coef(model_full$finalModel, model_full$bestTune$lambda))
coef_abs <- abs(coef_Ridge[-1, , drop = FALSE])  # Remove intercepto


#---------------------------------------------------
# Avaliar diferentes quantidades de variáveis
#---------------------------------------------------
feature_counts <- seq(5, min(50, nrow(coef_abs)), 5)
auc_scores <- c()
auc_log <- data.frame(N = integer(), AUC = numeric())

for (n in feature_counts) {
  vars_n <- names(sort(coef_abs[,1], decreasing = TRUE))[1:n]
  
  model_n <- train(x = X[, vars_n], y = Y,
                   method = "glmnet",
                   metric = "ROC",
                   trControl = ctrl)
  
  auc_val <- getTrainPerf(model_n)$TrainROC
  auc_scores <- c(auc_scores, auc_val)
  
  # Armazena os resultados no log
  auc_log <- rbind(auc_log, data.frame(N = n, AUC = auc_val))
}

#---------------------------------------------------
# Selecionar número ideal de variáveis
#---------------------------------------------------
best_n <- feature_counts[which.max(auc_scores)]
cat("Número ótimo de variáveis:", best_n, "\n")

#---------------------------------------------------
# Treinar modelo final com variáveis otimizadas
#---------------------------------------------------
top_features_ridge <- names(sort(coef_abs[,1], decreasing = TRUE))[1:best_n]

model_Ridge <- train(x = X[, top_features_ridge], y = Y,
                     method = "glmnet",
                     metric = "ROC",
                     trControl = ctrl)

#---------------------------------------------------
# Avaliação no conjunto de teste
#---------------------------------------------------
X_test <- testing[, top_features_ridge]
Y_test <- testing$TravelInsurance

pred <- predict(model_Ridge, newdata = X_test)
probs <- predict(model_Ridge, newdata = X_test, type = "prob")[, "S"]

roc_obj <- roc(Y_test, probs)
auc <- auc(roc_obj)

conf_matrix <- confusionMatrix(pred, Y_test)
precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

#---------------------------------------------------
# Resultados
#---------------------------------------------------
cat("AUC (teste):", round(auc, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-score:", round(f1_score, 4), "\n")
cat("Variáveis selecionadas:\n")
print(top_features_ridge)

#---------------------------------------------------
# Curva ROC
#---------------------------------------------------
plot(roc_obj, main = paste("ROC Curve - AUC:", round(auc, 2)))

#---------------------------------------------------
# Salvando os outputs
#---------------------------------------------------
save(model_Ridge, precision, recall, f1_score, auc, top_features_ridge, 
     file = "Ridge_model_evaluation.RData")



#---------------------------------------------------
# Grafico AUC top features
#---------------------------------------------------
# Identify optimal point
best_row <- auc_log[which.max(auc_log$AUC), ]

# Plot
ggplot(auc_log, aes(x = N, y = AUC)) +
  geom_line(color = "orange", linewidth = 1.2) +
  geom_point(color = "orange", size = 2) +
  geom_vline(xintercept = best_row$N, linetype = "dashed", color = "red", linewidth = 1) +
  geom_point(aes(x = best_row$N, y = best_row$AUC), color = "red", size = 3) +
  annotate("text", x = best_row$N, y = best_row$AUC + 0.002,
           label = paste("Optimal:", best_row$N, "variables"),
           color = "red", hjust = -0.1, size = 4.5) +
  labs(title = "AUC vs. Number of Selected Variables",
       x = "Number of Selected Variables",
       y = "AUC (cross-validation)") +
  theme_minimal(base_size = 14)

# (Optional) save the plot
# ggsave("AUC_vs_NumberVariables.png", width = 8, height = 6, dpi = 300)

# (Opcional) salvar o gráfico
# ggsave("AUC_vs_NumeroVariaveis.png", width = 8, height = 6, dpi = 300)



#---------------------------------------------------
# Matriz de confusão 
#---------------------------------------------------
# Previsão no conjunto de teste
pred <- predict(model_Ridge, newdata = X_test)

# Matriz de confusão
conf_matrix <- confusionMatrix(pred, Y_test, positive = "S")

# Exibir
print(conf_matrix)


#---------------------------------------------------
# TOP FEATURES
# Extrair coeficientes reais do modelo Ridge final
coef_ridge_final <- as.matrix(coef(model_Ridge$finalModel, model_Ridge$bestTune$lambda))

# Remover intercepto
coef_df <- data.frame(Variable = rownames(coef_ridge_final)[-1],
                      Coef = coef_ridge_final[-1, 1])

# Ordenar pelos valores absolutos para pegar os mais relevantes
n_top <- 10  # Número de variáveis a exibir
coef_df$AbsCoef <- abs(coef_df$Coef)
coef_top <- coef_df[order(-coef_df$AbsCoef), ][1:n_top, ]

# Classificação das cores por sinal
coef_top$Sign <- ifelse(coef_top$Coef >= 0, "Positive", "Negative")

# Reordenar fator para gráfico ordenado (mantendo sinal)
coef_top$Variable <- factor(coef_top$Variable, levels = coef_top$Variable[order(coef_top$Coef)])

# Plot
library(ggplot2)
ggplot(coef_top, aes(x = Coef, y = Variable, fill = Sign)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "steelblue", "Negative" = "firebrick")) +
  geom_text(aes(label = round(Coef, 3)), 
            hjust = ifelse(coef_top$Coef > 0, -0.1, 1.1), size = 4.2) +
  labs(
    title = paste("Top", n_top, "Coefficients of the Ridge Model"),
    x = "Coefficient (with sign)",
    y = "Variable",
    fill = "Sign"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  ) +
  coord_cartesian(xlim = c(min(coef_top$Coef) * 1.3, max(coef_top$Coef) * 1.3))
