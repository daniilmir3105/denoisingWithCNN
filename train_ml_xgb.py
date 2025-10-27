import numpy as np
import os
import pickle
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tqdm import tqdm  # для отображения прогрессбара


# Задайте пути к исходным файлам с сейсмограммами
# current_dir = r'/content/drive/MyDrive/datasets/seismic/denoising'
current_dir = os.path.dirname(os.path.abspath(__file__))
clean_spectra_path = os.path.join(current_dir, "spectres", "clean_spectra.npy")
noisy_spectra_path = os.path.join(current_dir, "spectres", "noisy_spectra.npy")

# Загружаем массивы с использованием memory mapping
X = np.load(noisy_spectra_path, mmap_mode='r')
Y = np.load(clean_spectra_path, mmap_mode='r')

# Определяем размеры выборок по срезам (80% train, 10% val, 10% test)
n = X.shape[0]
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
n_test  = n - n_train - n_val

X_train = X[:n_train]
X_val   = X[n_train:n_train + n_val]
X_test  = X[n_train + n_val:]

Y_train = Y[:n_train]
Y_val   = Y[n_train:n_train + n_val]
Y_test  = Y[n_train + n_val:]

print("Train set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])

# Фиксированные гиперпараметры модели
n_boost_rounds = 100
max_depth = 10
learning_rate = 0.1

# Инициализируем переменную для модели (будет накапливаться бустер)
model = None

# Создаем progressbar для отслеживания количества бустинговых итераций
progress_bar = tqdm(total=n_boost_rounds, desc="Обучение XGBRegressor")

# Инкрементальное обучение: на каждой итерации добавляем одну новую бустинговую итерацию
for i in range(n_boost_rounds):
    xgb = XGBRegressor(n_estimators=1,
                       max_depth=max_depth,
                       learning_rate=learning_rate,
                       verbosity=0,
                       random_state=42,
                       n_jobs=-1)  # n_jobs обеспечивает параллелизм внутри модели
    if model is not None:
        # Передаем предыдущий бустер для продолжения обучения
        xgb.fit(X_train, Y_train, xgb_model=model.get_booster())
    else:
        xgb.fit(X_train, Y_train)
    model = xgb
    progress_bar.update(1)
progress_bar.close()

# Оцениваем качество модели
mse_train = mean_squared_error(Y_train, model.predict(X_train))
mse_val   = mean_squared_error(Y_val, model.predict(X_val))
mse_test  = mean_squared_error(Y_test, model.predict(X_test))

print(f"XGBRegressor - Train MSE: {mse_train:.6f}, Val MSE: {mse_val:.6f}, Test MSE: {mse_test:.6f}")

if __name__ == "__main__":
    # pass
    # Сохраняем обученную модель в файл
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Лучшая модель (XGBRegressor) сохранена в файл best_model.pkl")
