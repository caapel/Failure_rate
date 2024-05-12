import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
X_test, y_test = make_regression(n_samples=100, n_features=1, noise=0.1)

# Инициализация пайплайнов с моделями
models = {
    "Gradient Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR())
    ]),
    "MLP": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(random_state=42, max_iter=500))
    ])
}

# Настройка кросс-валидации
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Оценка моделей с помощью кросс-валидации
metrics = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
scores = {}
for name, model in models.items():
    cv_results = cross_validate(model, X_test, y_test, cv=kf, scoring=metrics)
    scores[name] = {
        'MAE': -cv_results['test_neg_mean_absolute_error'].mean(),
        'MSE': -cv_results['test_neg_mean_squared_error'].mean(),
        'RMSE': -cv_results['test_neg_root_mean_squared_error'].mean()
    }

# Вывод результатов
for name, score_dict in scores.items():
    print(f'{name}:')
    for score_name, value in score_dict.items():
        print(f'  {score_name}: {value:.3f}')

# Визуализация результатов
n_groups = len(scores)
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

fig, ax = plt.subplots()
for i, metric in enumerate(['MAE', 'MSE', 'RMSE']):
    results = [scores[model][metric] for model in models]
    ax.bar(index + i * bar_width, results, bar_width, alpha=opacity, label=metric)

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of MAE, MSE, RMSE across Models')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(models.keys())
ax.legend()

plt.tight_layout()
plt.show()

from sklearn.inspection import permutation_importance

# Обучение моделей и вывод важности признаков для поддерживающих это моделей
feature_importances = {}
for name, pipeline in models.items():
    # Требуется обучить модель, если это не было сделано в рамках кросс-валидации
    model = pipeline.fit(X_test, y_test)
    if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        # Прямой доступ к важности признаков для моделей основанных на деревьях
        importances = model.named_steps['regressor'].feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_importances[name] = importances
        print(f"{name} feature importances:")
        for f in range(X_test.shape[1]):
            print(f"  {f + 1}. feature {indices[f]} ({importances[indices[f]]:.3f})")
    else:
        # Для других моделей, используем метод permutation_importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        feature_importances[name] = result.importances_mean
        indices = np.argsort(result.importances_mean)[::-1]
        print(f"{name} permutation importances:")
        for f in range(X_test.shape[1]):
            print(f"  {f + 1}. feature {indices[f]} ({result.importances_mean[indices[f]]:.3f})")

# Визуализация важности признаков
fig, ax = plt.subplots()
for i, (name, importances) in enumerate(feature_importances.items()):
    ax.bar(np.arange(len(importances)) + i * bar_width, importances, bar_width, label=name, alpha=opacity)

ax.set_xlabel('Feature indices')
ax.set_ylabel('Importance')
ax.set_title('Feature importances by Model')
ax.legend()
plt.show()

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# Генерация данных с 4 признаками
x_test, y_test = make_regression(n_samples=100, n_features=4, noise=0.1)

# Инициализация пайплайнов с моделями
models = {
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR())
    ]),
    "MLP": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(random_state=42, max_iter=500))
    ])
}

# Настройка кросс-валидации
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Оценка моделей с помощью кросс-валидации
metrics = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
scores = {}
feature_importances = {}

for name, model in models.items():
    if "Pipeline" in str(type(model)):
        # Для SVM и MLP не вычисляем важность признаков
        model.fit(x_test, y_test)
    else:
        # Обучение и получение важности признаков для моделей, основанных на деревьях
        model.fit(x_test, y_test)
        importances = model.feature_importances_
        feature_importances[name] = importances

    cv_results = cross_validate(model, x_test, y_test, cv=kf, scoring=metrics)
    scores[name] = {
        'MAE': -cv_results['test_neg_mean_absolute_error'].mean(),
        'MSE': -cv_results['test_neg_mean_squared_error'].mean(),
        'RMSE': -cv_results['test_neg_root_mean_squared_error'].mean()
    }

# Вывод важности признаков
for name, importances in feature_importances.items():
    print(f'{name} feature importances:')
    for idx, imp in enumerate(importances):
        print(f'  Feature {idx}: {imp:.3f}')

# Вывод результатов кросс-валидации
for name, score_dict in scores.items():
    print(f'{name}:')
    for score_name, value in score_dict.items():
        print(f'  {score_name}: {value:.3f}')