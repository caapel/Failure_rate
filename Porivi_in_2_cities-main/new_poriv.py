import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import plot_tree

# Загрузка данных
data = pd.read_excel('Poriv.xlsx', engine='openpyxl')
data2 = pd.read_excel('Poriv2.xlsx', engine='openpyxl')
common_columns = list(set(data.columns) & set(data2.columns))
data = data[common_columns].astype(str)
data2 = data2[common_columns].astype(str)
df_common = pd.merge(data, data2, how='outer', on=common_columns)

# Обработка и кодирование данных
df_common = df_common.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')

# Разделение данных
X = df_common.drop('Ki (действ)', axis=1)
y = df_common['Ki (действ)']

# Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Словарь для хранения пайплайнов
pipelines = {
    'ridge': Pipeline([('scaler', StandardScaler()), ('regressor', Ridge())]),
    'lasso': Pipeline([('scaler', StandardScaler()), ('regressor', Lasso())]),
    'elastic_net': Pipeline([('scaler', StandardScaler()), ('regressor', ElasticNet())]),
    'random_forest': Pipeline([('regressor', RandomForestRegressor(random_state=42))]),
    'gradient_boosting': Pipeline([('regressor', GradientBoostingRegressor(random_state=42))])
}

# Параметры для GridSearchCV
params = {
    'ridge': {'regressor__alpha': [0.01, 0.1, 1, 10, 100]},
    'lasso': {'regressor__alpha': [0.01, 0.1, 1, 10, 100]},
    'elastic_net': {'regressor__alpha': [0.01, 0.1, 1, 10, 100], 'regressor__l1_ratio': [0.1, 0.5, 0.9]},
    'random_forest': {'regressor__n_estimators': [10, 50, 100, 200]},
    'gradient_boosting': {'regressor__n_estimators': [100, 200, 300], 'regressor__learning_rate': [0.01, 0.1, 0.2]}
}

# Обучение и вывод результатов с обработкой исключений
for name, pipeline in pipelines.items():
    try:
        grid = GridSearchCV(pipeline, params[name], cv=5, scoring='neg_mean_squared_error', error_score='raise')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
        print(f"Best MSE for {name}: {-grid.best_score_}")

        if name in ['ridge', 'lasso', 'elastic_net']:
            print(f"Weights for {name}: {best_model.named_steps['regressor'].coef_}")
        elif name in ['random_forest', 'gradient_boosting']:
            importances = best_model.named_steps['regressor'].feature_importances_
            plot_feature_importances(importances, X.columns)
    except Exception as e:
        print(f"Error fitting model {name}: {e}")