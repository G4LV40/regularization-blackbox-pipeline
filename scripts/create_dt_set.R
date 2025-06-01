library(dplyr)
library(caret)
library(tidymodels)
library(themis)
library(fastDummies)
library(DMwR2)
library(smotefamily)

# Load the CSV file
dados <- read.csv("TravelInsurancePrediction.csv", header = TRUE)
dados <- dados[, 2:10]  # Drop the first column if it's just an index

# Convert to data frame
dados <- as.data.frame(dados)

# Check the class
class(dados)

# Feature engineering
dados <- dados %>%
  mutate(
    # 1. Family income per capita
    IncomePerCapita = AnnualIncome / FamilyMembers,
    
    # 2. High income indicator
    HighIncome = ifelse(AnnualIncome > quantile(AnnualIncome, 0.75), 1, 0),
    
    # 3. Normalized age
    AgeNormalized = scale(Age),
    
    # 4. High chronic diseases indicator
    HighChronicDiseases = ifelse(ChronicDiseases > median(ChronicDiseases), 1, 0),
    
    # 5. Travel frequency indicator
    TravelFrequency = ifelse(FrequentFlyer == "Yes" | EverTravelledAbroad == "Yes", 1, 0),
    
    # 6. Binary private employment indicator
    PrivateEmployment = ifelse(Employment.Type == "Private Sector/Self Employed", 1, 0),
    
    # 7. Low family dependence indicator
    LowDependence = ifelse(FamilyMembers <= 3, 1, 0),
    
    # 8. Income-to-age ratio
    IncomeByAge = AnnualIncome / Age,
    
    # 9. Age group categorization
    AgeGroup = case_when(
      Age <= 25 ~ "young",
      Age <= 35 ~ "Adult_y",
      Age <= 50 ~ "H_age",
      TRUE ~ "Senior"
    ),
    
    # 10. High income + travel interaction
    HighIncomeTraveler = ifelse((EverTravelledAbroad == "Yes" | FrequentFlyer == "Yes") & AnnualIncome > mean(AnnualIncome), 1, 0),
    
    # 11. Top 10% income percentile
    HighIncome90 = ifelse(AnnualIncome >= quantile(AnnualIncome, 0.90), 1, 0),
    
    # 12. Normalized income per capita
    IncomePerCapitaNorm = scale(IncomePerCapita),
    
    # 13. International travel experience
    ExperiencedTraveler = ifelse(EverTravelledAbroad == "Yes", 1, 0),
    
    # 14. Large family indicator
    LargeFamily = ifelse(FamilyMembers > quantile(FamilyMembers, 0.75), 1, 0),
    
    # 15. Chronic diseases per age
    ChronicByAge = (ChronicDiseases / Age),
    
    # 16. Insurance propensity score
    InsuranceScore = (AnnualIncome * 0.3 + ChronicDiseases * 0.4 + TravelFrequency * 0.2 + ExperiencedTraveler * 0.1),
    
    # 17. Financial dependence ratio
    FinancialDependence = log(FamilyMembers / AnnualIncome),
    
    # 18. Travel score
    TravelScore = ifelse(FrequentFlyer == "Yes", 1, 0) + ifelse(EverTravelledAbroad == "Yes", 2, 0),
    
    # 19. Estimated work experience (based on age)
    WorkExperience = ifelse(GraduateOrNot == "Yes", Age - 22, Age - 18),
    
    # 20. Job stability indicator
    StableJob = ifelse(Employment.Type == "Government Sector", 1, 0),
    
    # 21. Adjusted travel frequency by income
    AdjustedTravelIncome = ifelse(FrequentFlyer == "Yes", AnnualIncome / mean(AnnualIncome), 0),
    
    # 22. Perceived risk score
    RiskScore = (ChronicDiseases * 0.6 + Age * 0.2 + TravelFrequency * 0.2),
    
    # 23. Normalized risk score
    RiskScoreNorm = scale(RiskScore),
    
    # 24. Simulated cluster score (use real clustering in production)
    ClusterScore = kmeans(select(dados, AnnualIncome, Age, PrivateEmployment), centers = 3)$cluster,
    
    # 25. Average insurance take-up per cluster
    ClusterInsuranceRate = ave(TravelInsurance, ClusterScore, FUN = mean),
    
    # 26. Moving average of insurance take-up by age group
    MovingAvgInsurance = ave(TravelInsurance, AgeGroup, FUN = mean)
  )

# View financial dependence
dados$FinancialDependence

# Create dummy variables for categorical features
dados <- dummy_cols(dados,
                    select_columns = c("AgeGroup", "Employment.Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"),
                    remove_first_dummy = TRUE,  # Avoid multicollinearity
                    remove_selected_columns = TRUE  # Remove original categorical columns
)

# Rename dummy column for consistency
colnames(dados)[colnames(dados) == "Employment.Type_Private Sector/Self Employed"] <- "Employment_Type"

# Check label proportion (imbalance)
table(dados$TravelInsurance)[2] / dim(dados)[1]
table(dados$TravelInsurance)[1] / dim(dados)[1]

# Recode target variable as factors (S = Yes, N = No)
dados <- dados %>%
  mutate(TravelInsurance = recode(TravelInsurance, `0` = "N", `1` = "S"))

# Check structure
str(dados)

# Separate predictors (X) and target variable (Y)
X <- dados[, !colnames(dados) %in% "TravelInsurance"]
Y <- dados$TravelInsurance

# Apply SMOTE to balance the dataset
dados_smote <- SMOTE(X, Y, K = 5, dup_size = 1)

# Rebuild the dataset with balanced target
dados_balanciados <- dados_smote$data

# Rename 'class' back to 'TravelInsurance'
colnames(dados_balanciados)[colnames(dados_balanciados) == "class"] <- "TravelInsurance"

# Check new class distribution
table(dados_balanciados$TravelInsurance)

# Final data transformations
dados$TravelInsurance <- as.factor(dados$TravelInsurance)
dados$ChronicByAge <- log(dados$ChronicByAge)

# Final dataset for modeling
df <- dados_balanciados
str(df)

# Save the final dataset as .RData file
save(df, file = "TravelInsurancePrediction.RData")
