# Health Insurance Cross Sell Prediction
<img src="assets/shutterstock_373492012-health-insurance-e1491415000969.jpg" alt="Heal Insurance" width="600" height="300">

# 1. Description
  Esse projeto tem o objetivo de treinar um modelo de classificação binária para prever se clientes tem interesse em seguro de veículos de uma empresa utilizando dados do ano anterior.
  Esse estudo é de extrema importância, pois pode ajudar a empresa a planejar suas estratégias de comunicação para alcançar clientes e otimizar seu modelo de negócios e receita.

  ## Descrição dos dados

  | Variável | Definição |
  | -------- | -------- |
  | id  | Id exclusivo do cliente |
  | Gender  | Gênero do cliente |
  | Age  | Idade do Cliente |
  | Driving_License  | Se o cliente possui carteira de habilitação |
  | Region_Code  | Código exclusivo para a região do cliente |
  | Previously_Insured  | Se o cliente já tem seguro  |
  | Vehicle_Age  | Idade de Veículo |
  | Vehicle_Damage  | Se o cliente já teve seu veículo danificado no passado |
  | Annual_Premium  | O valor  que o cliente precisa pagar como prêmio |
  | Policy_Sales_Channel  | Código anonimizado para o canal de divulgação do cliente |
  | Vintage  | Número de dias que o cliente está associado a empresa |
  | Response  | Variável resposta se o cliente está ou não interessado na recontratação |
   

  1- Overview of the project
  2- what is about
  3- what it aims to achieve
  4- installation instructions

# 2. Analysis
## Overview
- The dataset contains **381,109 entries** and **10 columns**;
- The target variable **Response** is imbalanced, with approximately 12% positive classes;
- There are **no missing values** in the dataset;
-  **Numerical Features:** `Age`, `Annual_Premium` and `Vintage`;
-  **Categorical Features:** Variables `Region_Code` and `Policy_Sales_Channel` has many unique values;
-  **Low-variance Features:** `Driving_License` has **99.8%** of its values equal to 1;
- The variable **Policy_Sales_Channel** has 3 concentrated distributions of its categorical values.
- The **Annnual_Premium** distribution is positively skewed, with the value **$2360.0** ocurring at approximately 17% outside the main distribution
- **Region_ Code** distribution is varied, with 28 being the most common code. Most other codes have very few observations;

  ## Insights
  - Perform oversampling using **SMOTE**;
  - Apply **StandardScaler** to Numerical Features;
  - Use an appropriate **Enconding Strategy** to `Region_Code` and `Policy_Sales_Channel`, since one-hot encoding may not be suitable;
  - `Driving_License` can`be **dropped** without impacting the target variable;
  - Transform `Policy_Sales_Channel` by **clustering** them intom 3 groups;
  - Handle outliers and consider a **transformation** for the variable `Annual_Premium`;
  - **Handle** `Region_Code` by keeping the top 10 codes and grouping the other into an `Other` category;
  


# 3. Cross Validation
# 4. Hyperparameter Tuning
# 5. Final Evaluation
# 6. Business Insights
# 7. Refereneces
[Data](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

fastapi
kaggle
