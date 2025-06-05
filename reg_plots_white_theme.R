#---------------------------------------------------
# Libraries
#---------------------------------------------------
library(ggplot2)
library(dplyr)

coef_df <- data.frame(
  Variable = c(
    "MovingAvgInsurance", 
    "HighIncome90", 
    "HighIncome", 
    "ChronicDiseases",
    "AgeGroup_Young", 
    "ExperiencedTraveler", 
    "EverTravelledAbroad_Yes",
    "AdjustedTravelIncome", 
    "ClusterInsuranceRate", 
    "ChronicByAge"
  ),
  Coef = c(
    1.588, 
    1.442, 
    1.452, 
    0.595, 
    0.687, 
    0.479, 
    0.464, 
    0.494, 
    -2.894, 
    -33.415
  )
)



#---------------------------------------------------
# Add classification and order
#---------------------------------------------------
coef_df$AbsCoef <- abs(coef_df$Coef)
coef_df$Sign <- ifelse(coef_df$Coef >= 0, "Positive", "Negative")
coef_df$Variable <- factor(coef_df$Variable, levels = coef_df$Variable[order(coef_df$Coef)])

#---------------------------------------------------
# Plot
#---------------------------------------------------
plot_ <- ggplot(coef_df, aes(x = Coef, y = Variable, fill = Sign)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "steelblue", "Negative" = "firebrick")) +
  geom_text(aes(label = round(Coef, 3)), 
            hjust = ifelse(coef_df$Coef > 0, -0.1, 1.1), size = 4.2) +
  labs(
    title = "TOP 10 Coefficients of the Ridge Model",
    x = "Coefficients",
    y = "Variable",
    fill = "Sign"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray80"),
    panel.grid.minor = element_blank(),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black", face = "bold"),
    plot.title = element_text(face = "bold", color = "black"),
    legend.text = element_text(color = "black"),
    legend.title = element_text(color = "black"),
    legend.position = "bottom"
  ) +
  coord_cartesian(xlim = c(min(coef_df$Coef) * 1.3, max(coef_df$Coef) * 1.3))

# Salvar imagem final
ggsave("top_10_Ridge.png", plot = plot_, dpi = 800, width = 8.5, height = 6, units = "in")


#-------------------------------------------------------------------------------


coef_df <- data.frame(
  Variable = c(
    "MovingAvgInsurance", 
    "HighIncome90", 
    "HighIncome", 
    "ExperiencedTraveler", 
    "AgeGroup_Young", 
    "HighIncomeTraveler", 
    "LowDependence", 
    "RiskScore", 
    "LargeFamily", 
    "ClusterInsuranceRate"
  ),
  Coef = c(
    2.610, 
    1.446, 
    0.937, 
    0.743, 
    0.288, 
    0.288, 
    0.237, 
    0.216, 
    0.167, 
    -0.798
  )
)


#---------------------------------------------------
# Add classification and order
#---------------------------------------------------
coef_df$AbsCoef <- abs(coef_df$Coef)
coef_df$Sign <- ifelse(coef_df$Coef >= 0, "Positive", "Negative")
coef_df$Variable <- factor(coef_df$Variable, levels = coef_df$Variable[order(coef_df$Coef)])

#---------------------------------------------------
# Plot
#---------------------------------------------------
plot_ <- ggplot(coef_df, aes(x = Coef, y = Variable, fill = Sign)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "steelblue", "Negative" = "firebrick")) +
  geom_text(aes(label = round(Coef, 3)), 
            hjust = ifelse(coef_df$Coef > 0, -0.1, 1.1), size = 4.2) +
  labs(
    title = "TOP 10 Coefficients of the LASSO Model",
    x = "Coefficients",
    y = "Variable",
    fill = "Sign"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray80"),
    panel.grid.minor = element_blank(),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black", face = "bold"),
    plot.title = element_text(face = "bold", color = "black"),
    legend.text = element_text(color = "black"),
    legend.title = element_text(color = "black"),
    legend.position = "bottom"
  ) +
  coord_cartesian(xlim = c(min(coef_df$Coef) * 1.3, max(coef_df$Coef) * 1.3))


# Salvar imagem final
ggsave("top_10_LASSO.png", plot = plot_, dpi = 800, width = 8.5, height = 6, units = "in")
#-------------------------------------------------------------------------------

coef_df <- data.frame(
  Variable = c(
    "MovingAvgInsurance", 
    "HighIncome90", 
    "HighIncome", 
    "AgeGroup_Young", 
    "EverTravelledAbroad_Yes", 
    "Experiencedtraveler",
    "ClusterScore",
    "TravelFrequency", 
    "ClusterInsuranceRate", 
    "ChronicByAge"
  ),
  Coef = c(
    1.678, 
    1.460, 
    1.023, 
    0.494, 
    0.399, 
    0.394, 
    0.243,
    -0.248,
    -1.911, 
    -9.889
  )
)

#---------------------------------------------------
# Add classification and order
#---------------------------------------------------
coef_df$AbsCoef <- abs(coef_df$Coef)
coef_df$Sign <- ifelse(coef_df$Coef >= 0, "Positive", "Negative")
coef_df$Variable <- factor(coef_df$Variable, levels = coef_df$Variable[order(coef_df$Coef)])

#---------------------------------------------------
# Plot
#---------------------------------------------------
plot_ <- ggplot(coef_df, aes(x = Coef, y = Variable, fill = Sign)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "steelblue", "Negative" = "firebrick")) +
  geom_text(aes(label = round(Coef, 3)), 
            hjust = ifelse(coef_df$Coef > 0, -0.1, 1.1), size = 4.2) +
  labs(
    title = "TOP 10 Coefficients of the Elastic Net Model",
    x = "Coefficients",
    y = "Variable",
    fill = "Sign"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray80"),
    panel.grid.minor = element_blank(),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black", face = "bold"),
    plot.title = element_text(face = "bold", color = "black"),
    legend.text = element_text(color = "black"),
    legend.title = element_text(color = "black"),
    legend.position = "bottom"
  ) +
  coord_cartesian(xlim = c(min(coef_df$Coef) * 1.3, max(coef_df$Coef) * 1.3))


# Salvar imagem final
ggsave("top_10_ElasticNet.png", plot = plot_, dpi = 800, width = 8.5, height = 6, units = "in")


library(ggpubr)


# Combined
plot_Ridge <- plot_Ridge + labs(title = "(a) Ridge")
plot_LASSO <- plot_LASSO + labs(title = "(b) Lasso")
plot_EN    <- plot_EN + labs(title = "(c) Elastic Net")

top_10_combined <- ggarrange(
  plot_Ridge, plot_LASSO, plot_EN,
  ncol = 1, nrow = 3,
  align = "v",
  heights = c(1.1, 1.1, 1.1)
)

ggsave("top_10_combined_clean.png", plot = top_10_combined, 
       dpi = 600, width = 9, height = 14, units = "in")


