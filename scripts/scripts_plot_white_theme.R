library(tidyverse)
library(ggplot2)
library(dplyr)

load("results_simulation.RData")

ridge_index <- grep("Ridge", df_results$model)
lasso_index <- grep("Lasso", df_results$model)
en_index <- grep("ElasticNet", df_results$model)
full_index <- grep("Full", df_results$model)

reg_type <- character(nrow(df_results))
reg_type[ridge_index] <- "Ridge"
reg_type[lasso_index] <- "Lasso"
reg_type[en_index] <- "ElasticNet"
reg_type[full_index] <- "No reg."

rf_index <- grep("RF", df_results$model)
xgb_index <- grep("XGBoost", df_results$model)
lgbm_index <- grep("LightGBM", df_results$model)
cb_index <- grep("CatBoost", df_results$model)
h2o_index <- grep("H2OGBM", df_results$model)

model_type <- character(nrow(df_results))
model_type[rf_index] <- "RF"
model_type[xgb_index] <- "XGBoost"
model_type[lgbm_index] <- "LightGBM"
model_type[cb_index] <- "CatBoost"
model_type[h2o_index] <- "H2OGBM"
model_type[model_type == ""] <- "Reg. only"

reg_type <- factor(reg_type, levels = c("No reg.","Ridge","Lasso","ElasticNet"))
model_type <- factor(model_type, levels = c("Reg. only","RF","XGBoost","LightGBM","CatBoost","H2OGBM"))

df_results %>%
  mutate(p = factor(p),
         reg_type = reg_type,
         model_type = model_type) %>%
  ggplot(aes(x = p, y = rmse)) +
  theme_bw() +
  geom_boxplot() +
  facet_grid(model_type ~ reg_type)

df_results %>%
  mutate(p = factor(p),
         reg_type = reg_type,
         model_type = model_type) %>%
  ggplot(aes(y = rmse, fill = model_type)) +
  theme_bw() +
  geom_boxplot() +
  facet_grid(p ~ reg_type) +
  scale_x_continuous(labels = NULL)

df_results %>%
  mutate(p = factor(p, levels = c(5,10,50),
                    labels = c("5 true cov.","5 true cov. + 5 noise cov.","5 true cov. + 45 noise cov.")),
         reg_type = reg_type,
         model_type = model_type) %>%
  ggplot(aes(x = model_type, y = rmse, fill = reg_type)) +
  theme_bw() +
  geom_boxplot(outlier.size = .5, lwd = .3) +
  facet_wrap(~ p, ncol = 1) +
  xlab("Model type") +
  scale_fill_discrete(name = "Reg. type") +
  ylab("RMSE")

my_plot <- df_results %>%
  mutate(p = factor(p, levels = c(5,10,50),
                    labels = c("5 true cov.","5 true cov. + 5 noise cov.","5 true cov. + 45 noise cov.")),
         reg_type = reg_type,
         model_type = model_type) %>%
  ggplot(aes(x = model_type, y = rmse, fill = reg_type)) +
  theme_bw() +
  geom_boxplot(outlier.size = .5, lwd = .3) +
  facet_wrap(~ p, ncol = 1) +
  xlab("Model type") +
  scale_fill_discrete(name = "Reg. type") +
  ylab("RMSE")


#-------------------------------------------------------------------------------
# Graphics
#-------------------------------------------------------------------------------



# Define the plot function for each n
plot_by_sample_size <- function(current_n) {
  df_results %>%
    filter(n == current_n) %>%
    mutate(
      p = factor(p, levels = c(5, 10, 50),
                 labels = c("5 true cov.", "5 true cov. + 5 noise cov.", "5 true cov. + 45 noise cov.")),
      reg_type = factor(case_when(
        grepl("Ridge", model) ~ "Ridge",
        grepl("Lasso", model) ~ "Lasso",
        grepl("ElasticNet", model) ~ "ElasticNet",
        TRUE ~ "No reg."
      ), levels = c("No reg.", "Ridge", "Lasso", "ElasticNet")),
      model_type = factor(case_when(
        grepl("RF", model) ~ "RF",
        grepl("XGBoost", model) ~ "XGBoost",
        grepl("LightGBM", model) ~ "LightGBM",
        grepl("CatBoost", model) ~ "CatBoost",
        grepl("H2OGBM", model) ~ "H2OGBM",
        TRUE ~ "Reg. only"
      ), levels = c("Reg. only", "RF", "XGBoost", "LightGBM", "CatBoost", "H2OGBM"))
    ) %>%
    ggplot(aes(x = model_type, y = rmse, fill = reg_type)) +
    geom_boxplot(outlier.size = .5, lwd = .3) +
    facet_wrap(~ p, ncol = 1) +
    scale_fill_manual(values = c(
      "No reg." = "#E41A1C", "Ridge" = "#4DAF4A",
      "Lasso" = "#377EB8", "ElasticNet" = "#984EA3"
    )) +
    theme_bw(base_size = 12) +
    labs(
      title = paste("RMSE by Model Type and Regularization - Sample size =", current_n),
      x = "Model type",
      y = "RMSE",
      fill = "Reg. type"
    )
}

# Loop to create and save plots for each sample size
for (n_val in unique(df_results$n)) {
  plot_n <- plot_by_sample_size(n_val)
  print(plot_n)  # if running in interactive session
  
  ggsave(
    filename = paste0("rmse_plot_n_", n_val, ".png"),
    plot = plot_n,
    dpi = 800, width = 6, height = 6
  )
}

ggsave(filename = "plot_luciano.png", plot = my_plot, dpi = 800,
       width = 6, height = 6)





