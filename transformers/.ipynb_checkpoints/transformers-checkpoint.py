import numpy as np

class PeakValueTransformer():
    def __init__(self, peak_value, method='median'):
        self.peak_value = peak_value
        self.method = method
        self.replacement_ = None
    
    def fit(self, X, y=None):
        # X é um DataFrame com uma coluna
        if self.method == 'median':
            self.replacement_ = X.iloc[:,0].median()
        elif self.method == 'mean':
            self.replacement_ = X.iloc[:,0].mean()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        col_name = X_transformed.columns[0]
        
        # Criar coluna flag do pico
        flag_col_name = col_name + '_flag'
        X_transformed[flag_col_name] = (X_transformed[col_name] == self.peak_value).astype(int)
        
        # Substituir o valor do pico na coluna original
        X_transformed.loc[X_transformed[col_name] == self.peak_value, col_name] = self.replacement_
        
        return X_transformed



        

class IQROutlierRemover():
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        self.limite_inferior_ = Q1 - self.factor * (Q3 - Q1)
        self.limite_superior_ = Q3 + self.factor * (Q3 - Q1)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = X_transformed[col].clip(
                lower=self.limite_inferior_[col],
                upper=self.limite_superior_[col]
            )
        return X_transformed




















# # class MeuTransformador(BaseEstimator, TransformerMixin):
#     def __init__(self, parametro1=valor1):
#         # Aqui você define parâmetros que podem ser alterados ao criar o objeto
#         self.parametro1 = parametro1

#     def fit(self, X, y=None):
#         # Aqui você "aprende" coisas do dataset de treino
#         # Ex.: calcular média, mediana, quantis, valores únicos etc.
#         # Deve retornar self
#         return self

#     def transform(self, X):
#         # Aqui você aplica a transformação
#         # Pode criar cópia de X para não modificar original
#         X = X.copy()
#         # Ex.: substituir valores, criar colunas novas, normalizar etc.
#         return X