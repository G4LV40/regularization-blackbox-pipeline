
library(dplyr)
library(caret)
library(tidymodels)
library(themis)
library(fastDummies)
library(DMwR2)
library(smotefamily)


#setwd('/home/g4lv40/Desktop/Doutorado/Artigos_TESE/Modelos_Whitebox_R/2025_novo')

# Carregar o arquivo .rda
dados<-read.csv("TravelInsurancePrediction.csv", h=T)
dados<-dados[,2:10]
#load('TravelInsurancePrediction.RData')

# Estruturando o Dataset e Criando Dummies

dados<-as.data.frame(dados)

class(dados)
# Criando novas features
dados <- dados %>%
  mutate(
    # 1. Renda per capita da família
    IncomePerCapita = AnnualIncome / FamilyMembers,
    
    # 2. Indicador de alta renda
    HighIncome = ifelse(AnnualIncome > quantile(AnnualIncome, 0.75), 1, 0),
    
    # 3. Idade normalizada
    AgeNormalized = scale(Age),
    
    # 4. Doenças crônicas normalizadas
    HighChronicDiseases = ifelse(ChronicDiseases > median(ChronicDiseases), 1, 0),
    
    # 5. Frequência de viagem
    TravelFrequency = ifelse(FrequentFlyer == "Yes" | EverTravelledAbroad == "Yes", 1, 0),
    
    # 6. Emprego no setor privado (binário)
    PrivateEmployment = ifelse(Employment.Type == "Private Sector/Self Employed", 1, 0),
    
    # 7. Indicador de baixa dependência familiar
    LowDependence = ifelse(FamilyMembers <= 3, 1, 0),
    
    # 8. Razão entre renda e idade
    IncomeByAge = AnnualIncome / Age,
    
    # 9. Faixa etária categorizada
    AgeGroup = case_when(
      Age <= 25 ~ "young",
      Age <= 35 ~ "Adult_y",
      Age <= 50 ~ "H_age",
      TRUE ~ "Senior"
    ),
    
    # 10. Interação entre viagem e renda
    HighIncomeTraveler = ifelse((EverTravelledAbroad == "Yes" | FrequentFlyer == "Yes") & AnnualIncome > mean(AnnualIncome), 1, 0),
    
    # 11. Percentil de renda
    HighIncome90 = ifelse(AnnualIncome >= quantile(AnnualIncome, 0.90), 1, 0),
    
    # 12. Renda per capita normalizada
    IncomePerCapitaNorm = scale(IncomePerCapita),
    
    # 13. Status de experiência internacional
    ExperiencedTraveler = ifelse(EverTravelledAbroad == "Yes", 1, 0),
    
    # 14. Indicador de alta família dependente
    LargeFamily = ifelse(FamilyMembers > quantile(FamilyMembers, 0.75), 1, 0),
    
    # 15. Razão entre doenças crônicas e idade
    ChronicByAge = (ChronicDiseases / Age),
    
    # 16. Score de propensão ao seguro
    InsuranceScore = (AnnualIncome * 0.3 + ChronicDiseases * 0.4 + TravelFrequency * 0.2 + ExperiencedTraveler * 0.1),
    
    # 17. Razão de dependência financeira
    FinancialDependence = log(FamilyMembers / AnnualIncome),
    
    # 18. Score de viagem
    TravelScore = ifelse(FrequentFlyer == "Yes", 1, 0) + ifelse(EverTravelledAbroad == "Yes", 2, 0),
    
    # 19. Tempo de mercado estimado
    WorkExperience = ifelse(GraduateOrNot == "Yes", Age - 22, Age - 18),
    
    # 20. Indicador de estabilidade profissional
    StableJob = ifelse(Employment.Type == "Government Sector", 1, 0),
    
    # 21. Frequência de viagem ajustada pela renda
    AdjustedTravelIncome = ifelse(FrequentFlyer == "Yes", AnnualIncome / mean(AnnualIncome), 0),
    
    # 22. Score de risco percebido
    RiskScore = (ChronicDiseases * 0.6 + Age * 0.2 + TravelFrequency * 0.2),
    
    # 23. Score de risco percebido normalizado
    RiskScoreNorm = scale(RiskScore),
    
    # 24. Indicador de seguro por cluster (simulado, requer clusterização real)
    ClusterScore = kmeans(select(dados, AnnualIncome, Age, PrivateEmployment), centers = 3)$cluster,
    
    # 25. Probabilidade média de contratação por cluster
    ClusterInsuranceRate = ave(TravelInsurance, ClusterScore, FUN = mean),
    
    # 26. Média móvel de contratação de seguro por grupo similar (simulação)
    MovingAvgInsurance = ave(TravelInsurance, AgeGroup, FUN = mean)
  )

dados$FinancialDependence

# Criar dummies apenas para variáveis categóricas
dados <- dummy_cols(dados, select_columns = c("AgeGroup","Employment.Type","GraduateOrNot","FrequentFlyer","EverTravelledAbroad"), 
                 remove_first_dummy = TRUE,  # Evita colinearidade
                 remove_selected_columns = TRUE)  # Remove as colunas originais categóricas

colnames(dados)[colnames(dados) == "Employment.Type_Private Sector/Self Employed"] <- "Employment_Type"



#verificar a proporção da label para ( balanceado)
table(dados$TravelInsurance)[2]/dim(dados)[1]
table(dados$TravelInsurance)[1]/dim(dados)[1]


dados <- dados %>%
  mutate(TravelInsurance = recode(TravelInsurance, `0` = "N", `1` = "S"))



str(dados)
####Balenceado
#X: Conjunto de variáveis preditoras.
#Y: Vetor de labels.
#K = 5: Número de vizinhos usados para a interpolação SMOTE.
#dup_size = 2: Número de vezes que as amostras da classe minoritária serão replicadas. Ajuste conforme necessário.



#  Separar as variáveis preditoras (X) e a variável resposta (Y)
X <- dados[, !colnames(dados) %in% "TravelInsurance"]  # Todas as colunas exceto a target
Y <- dados$TravelInsurance  # Guardando a variável target

#  Aplicar o SMOTE
dados_smote <- SMOTE(X, Y, K = 5, dup_size = 1)  # dup_size controla a quantidade de oversampling

# Reconstruir o dataset com a variável target correta
dados_balanciados <- dados_smote$data


#  Renomear 'class' de volta para 'TravelInsurance' para manter o nome original
colnames(dados_balanciados)[colnames(dados_balanciados) == "class"] <- "TravelInsurance"

#  Conferir a nova distribuição das classes após o balanceamento
table(dados_balanciados$TravelInsurance)


dados$TravelInsurance<-as.factor(dados$TravelInsurance)
dados$ChronicByAge<-log(dados$ChronicByAge)

df<-dados_balanciados
str(df)

# Salvando os outputs dos modelos
save(df, file = "TravelInsurancePrediction.RData")
