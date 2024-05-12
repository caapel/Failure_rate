import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import openpyxl
from sklearn.tree import plot_tree

# Загрузка данных и предварительная обработка (необходимо адаптировать пути к файлам)
data = pd.read_excel('Poriv.xlsx', engine='openpyxl')
data2 = pd.read_excel('Poriv2.xlsx', engine='openpyxl')

common_columns = list(set(data.columns) & set(data2.columns))

for column in common_columns:
    data[column] = data[column].astype(str)
    data2[column] = data2[column].astype(str)

merged_data_by_common_columns = pd.merge(data, data2, how='outer', on=common_columns)

selected_columns = [
    "Утонение стенки, %",
    "Наличие других порывов на участке, К2",
    "Коррозионная активность грунта, К3",
    "Наличие/отсутствие затопления (следов затопления) канала, К4",
    "Наличие пересечений с коммуникациями, К5",
    "Ki (действ)"
]

df_common = merged_data_by_common_columns[selected_columns].copy()
# Превращаем колонки в числовой тип
df_common['Утонение стенки, %'] = pd.to_numeric(df_common['Утонение стенки, %'], errors='coerce')

df = df_common.dropna(subset=['Утонение стенки, %'], inplace=True)
df_common['Ki (действ)'] = pd.to_numeric(df_common['Ki (действ)'], errors='coerce')
# Кодируем строковые признаки
df_common['Наличие других порывов на участке, К2'] = df_common['Наличие других порывов на участке, К2'].astype('category')
df_common['Наличие других порывов на участке, К2'] = df_common['Наличие других порывов на участке, К2'].cat.codes
df_common['Коррозионная активность грунта, К3'] = df_common['Коррозионная активность грунта, К3'].astype('category')
df_common['Коррозионная активность грунта, К3'] = df_common['Коррозионная активность грунта, К3'].cat.codes
df_common['Наличие/отсутствие затопления (следов затопления) канала, К4'] = df_common['Наличие/отсутствие затопления (следов затопления) канала, К4'].astype('category')
df_common['Наличие/отсутствие затопления (следов затопления) канала, К4'] = df_common['Наличие/отсутствие затопления (следов затопления) канала, К4'].cat.codes
df_common['Наличие пересечений с коммуникациями, К5'] = df_common['Наличие пересечений с коммуникациями, К5'].astype('category')
df_common['Наличие пересечений с коммуникациями, К5'] = df_common['Наличие пересечений с коммуникациями, К5'].cat.codes
print(df_common['Наличие других порывов на участке, К2'])

for column in df_common.columns:
        df_common[column] = pd.to_numeric(df_common[column], errors='ignore')
for column in df_common.columns:
    if df_common[column].dtype in ['int64', 'float64'] and df_common[column].isnull().any():
        mean_value = df_common[column].mean()
        df_common.loc[:, column] = df_common[column].fillna(mean_value)
    elif df_common[column].dtype == 'object' and df_common[column].isnull().any():
        mode_value = df_common[column].mode()[0]
        df_common.loc[:, column] = df_common[column].fillna(mode_value)

for column in df_common.columns:
    if df_common[column].dtype == 'object':
        df_common.loc[:, column] = df_common[column].astype('category')

df_common = df_common.sort_values(by = 'Ki (действ)', ascending=True)
df_common.reset_index(drop= True, inplace=True)
df_common = df_common.iloc[5:85]
df_common.reset_index(drop= True, inplace=True)
print(df_common)

df_common = df_common.iloc[3:90]
df_common.reset_index(drop= True, inplace=True)
df_common

print(df_common.info())
print(df_common.shape)

corr_matrix = df_common.corr()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Матрица корреляции')
plt.show()
# Гистограммы для непрерывных признаков
df_common.hist(bins=15, figsize=(15, 10))
plt.suptitle('Гистограммы распределения признаков')
plt.show()

# Ящики с усами для категориальных признаков
plt.figure(figsize=(10, 6))
for index, feature in enumerate(["Утонение стенки, %", "Наличие других порывов на участке, К2", "Коррозионная активность грунта, К3", "Наличие/отсутствие затопления (следов затопления) канала, К4", "Наличие пересечений с коммуникациями, К5"]):
    plt.subplot(2, 3, index + 1)
    sns.boxplot(x=df_common[feature])
plt.tight_layout

X = df_common.drop('Ki (действ)'  , axis=1)  # 'y' замените на имя вашей целевой колонки
y = df_common['Ki (действ)']  # 'y' замените на имя вашей целевой колонки


scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(df_common)
# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Дополнительное разделение обучающего набора на обучающий и валидационный наборы
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


# Инициализация моделей
models = {
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVM": SVR(),
    "MLP": MLPRegressor(random_state=42, max_iter=500)
}

# Обучение моделей и предсказание
predictions = {}
errors = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_val)
    errors[name] = (predictions[name] - y_val) / y_val * 100  # Расчет относительной ошибки в процентах

# Визуализация ошибок
plt.figure(figsize=(10, 6))
markers = {'Gradient Boosting': 's', 'Random Forest': '^', 'SVM': 'o', 'MLP': 'd'}
colors = {'Gradient Boosting': 'blue', 'Random Forest': 'green', 'SVM': 'red', 'MLP': 'purple'}
for name, err in errors.items():
    plt.scatter(range(len(err)), err, marker=markers[name], color=colors[name], label=name)

plt.title('Сравнение относительных ошибок предсказаний моделей')
plt.xlabel('Номер точки в валидационном наборе')
plt.ylabel('Относительная ошибка (%)')
plt.legend()
plt.grid(True)
plt.show()
# Инициализация моделей
models = {
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Обучение моделей
for name, model in models.items():
    model.fit(X_train, y_train)
    # Вывод важности признаков
    print(f"{name} feature importances:", model.feature_importances_)
# Полный набор признаков
full_features = X_train

# Удаление одного наименее важного признака
reduced_features_1 = X_train[:, :-1]

# Удаление двух наименее важных признаков
reduced_features_2 = X_train[:, :-2]

# Сравнение производительности
feature_sets = {"Full set": full_features, "Without 1 feature": reduced_features_1, "Without 2 features": reduced_features_2}
results = {}

for name, model in models.items():
    results[name] = {}
    for set_name, X_feat in feature_sets.items():
        model.fit(X_feat, y_train)
        pred = model.predict(X_val[:, :X_feat.shape[1]])
        mse = mean_squared_error(y_val, pred)
        results[name][set_name] = mse
        print(f"{name} {set_name} MSE: {mse}")
import matplotlib.pyplot as plt

# Визуализация ошибок MSE для разных наборов признаков
plt.figure(figsize=(10, 6))
for name in models:
    mse_values = list(results[name].values())
    plt.plot(mse_values, marker='o', label=name)

plt.xticks(ticks=np.arange(len(feature_sets)), labels=feature_sets.keys())
plt.ylabel('MSE')
plt.title('Сравнение MSE при разном количестве признаков')
plt.legend()
plt.grid(True)
plt.show()