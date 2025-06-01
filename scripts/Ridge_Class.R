#---------------------------------------------------
# Required Libraries
#---------------------------------------------------
library(caret)
library(pROC)
library(glmnet)
library(ggplot2)

#---------------------------------------------------
# Initial Settings
#---------------------------------------------------
set.seed(42)

#---------------------------------------------------
# Data Loading
#---------------------------------------------------
load(file = "TravelInsurancePrediction.RData")
df_1 <- df
rm(df)
gc()

#---------------------------------------------------
# Preprocessing
#---------------------------------------------------
df_1 <- df_1[, -20]  # Remove unspecified column
df_1$TravelInsurance <- as.factor(df_1$TravelInsurance)

#---------------------------------------------------
# Train/Test Split
#---------------------------------------------------
inTraining <- createDataPartition(df_1$TravelInsurance, p = 0.80, list = FALSE)
training <- df_1[inTraining, ]
testing <- df_1[-inTraining, ]

#---------------------------------------------------
# Training Control Configuration
#---------------------------------------------------
ctrl <- trainControl(method = "cv",
                     number = 10,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     verboseIter = FALSE,
                     allowParallel = TRUE)

#---------------------------------------------------
# Ridge Model Training with All Variables
#---------------------------------------------------
X <- training[, -which(names(training) == "TravelInsurance")]
Y <- training$TravelInsurance

model_full <- train(x = X, y = Y,
                    method = "glmnet",
                    metric = "ROC",
                    trControl = ctrl)

#---------------------------------------------------
# Coefficients from the Best Ridge Model
#---------------------------------------------------
coef_Ridge <- as.matrix(coef(model_full$finalModel, model_full$bestTune$lambda))
coef_abs <- abs(coef_Ridge[-1, , drop = FALSE])  # Remove intercept

#---------------------------------------------------
# Evaluate Different Numbers of Variables
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
  
  # Store results in log
  auc_log <- rbind(auc_log, data.frame(N = n, AUC = auc_val))
}

#---------------------------------------------------
# Select Optimal Number of Variables
#---------------------------------------------------
best_n <- feature_counts[which.max(auc_scores)]
cat("Optimal number of variables:", best_n, "\n")

#---------------------------------------------------
# Final Ridge Model with Selected Variables
#---------------------------------------------------
top_features_ridge <- names(sort(coef_abs[,1], decreasing = TRUE))[1:best_n]

model_Ridge <- train(x = X[, top_features_ridge], y = Y,
                     method = "glmnet",
                     metric = "ROC",
                     trControl = ctrl)

#---------------------------------------------------
# Evaluation on the Test Set
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
# Evaluation Metrics
#---------------------------------------------------
cat("AUC (test):", round(auc, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-score:", round(f1_score, 4), "\n")
cat("Selected variables:\n")
print(top_features_ridge)

#---------------------------------------------------
# ROC Curve
#---------------------------------------------------
plot(roc_obj, main = paste("ROC Curve - AUC:", round(auc, 2)))

#---------------------------------------------------
# Saving Outputs
#---------------------------------------------------
save(model_Ridge, precision, recall, f1_score, auc, top_features_ridge, 
     file = "Ridge_model_evaluation.RData")

#---------------------------------------------------
# AUC vs. Number of Features Plot
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


#---------------------------------------------------
# Confusion Matrix
#---------------------------------------------------
# Predict on test set
pred <- predict(model_Ridge, newdata = X_test)

# Confusion matrix
conf_matrix <- confusionMatrix(pred, Y_test, positive = "S")

# Display
print(conf_matrix)

#---------------------------------------------------
# Top Features
# Extract real coefficients from final Ridge model
coef_ridge_final <- as.matrix(coef(model_Ridge$finalModel, model_Ridge$bestTune$lambda))

# Remove intercept
coef_df <- data.frame(Variable = rownames(coef_ridge_final)[-1],
                      Coef = coef_ridge_final[-1, 1])

# Order by absolute value to select most relevant variables
n_top <- 10  # Number of top features to display
coef_df$AbsCoef <- abs(coef_df$Coef)
coef_top <- coef_df[order(-coef_df$AbsCoef), ][1:n_top, ]

# Classify sign
coef_top$Sign <- ifelse(coef_top$Coef >= 0, "Positive", "Negative")

# Reorder factor for ordered plot
coef_top$Variable <- factor(coef_top$Variable, levels = coef_top$Variable[order(coef_top$Coef)])

# Plot
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
